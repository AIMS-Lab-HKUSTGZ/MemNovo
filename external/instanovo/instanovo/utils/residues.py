from __future__ import annotations

import re

import numpy as np
import torch

from instanovo.constants import H2O_MASS, PROTON_MASS_AMU, SpecialTokens


class ResidueSet:
    """A class for managing sets of residues.

    Args:
        residue_masses (dict[str, float]):
            Dictionary of residues mapping to corresponding mass values.
        residue_remapping (dict[str, str] | None, optional):
            Dictionary of residues mapping to keys in `residue_masses`.
            This is used for dataset specific residue naming conventions.
            Residue remapping may be many-to-one.
    """

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str] | None = None,
    ) -> None:
        self.residue_masses = residue_masses
        self.residue_remapping = residue_remapping if residue_remapping else {}

        # Special tokens come first
        self.special_tokens = [
            SpecialTokens.PAD_TOKEN.value,
            SpecialTokens.SOS_TOKEN.value,
            SpecialTokens.EOS_TOKEN.value,
        ]

        self.vocab = self.special_tokens + list(self.residue_masses.keys())

        # Create mappings
        self.residue_to_index = {residue: index for index, residue in enumerate(self.vocab)}
        self.index_to_residue = dict(enumerate(self.vocab))
        # Split on amino acids allowing for modifications eg. AM(ox)Z -> [A, M(ox), Z]
        # Supports brackets or unimod notation
        self.tokenizer_regex = (
            # First capture group: matches standalone modifications
            # These would represent n-terminal modifications:
            # - Anything in square brackets, like [UNIMOD:35], [GLY:123456], [+.98]
            # - Anything in parentheses, like (ox), (-17.02), (+.98), (p)
            # - Raw numeric modifications with optional sign, like +15.99, -17.02, 0.98, .98
            r"(\[[^\]]+\]"  # Square-bracketed mod, e.g. [UNIMOD:123]
            r"|\([^)]+\)"  # Parentheses mod, e.g. (+15.99)
            r"|[+-]?\d+(?:\.\d+)?"  # Number with optional +/-, e.g. +15.99, -17.02, 42.02
            r"|[+-]?\.\d+"  # Decimal without leading digit, e.g. .98
            r")|"
            # Second capture group: matches amino acids with optional attached modifications
            # - Starts with a capital letter A-Z (standard amino acid codes)
            # - Optionally followed by any of the above modification formats
            r"([A-Z]"  # Amino acid residue, e.g. A, R, C
            r"(?:\[[^\]]+\]"  # Optional square-bracketed mod
            r"|\([^)]+\)"  # Optional parentheses mod
            r"|[+-]?\d+(?:\.\d+)?"  # Optional numeric mod
            r"|[+-]?\.\d+)?"
            r")"
        )

        self.PAD_INDEX: int = self.residue_to_index[SpecialTokens.PAD_TOKEN.value]
        self.SOS_INDEX: int = self.residue_to_index[SpecialTokens.SOS_TOKEN.value]
        self.EOS_INDEX: int = self.residue_to_index[SpecialTokens.EOS_TOKEN.value]

    def update_remapping(self, mapping: dict[str, str]) -> None:
        """Update the residue remapping for specific datasets.

        Args:
            mapping (dict[str, str]):
                The mapping from residues specific to a dataset
                to residues in the original `residue_masses`.
        """
        self.residue_remapping.update(mapping)

    def get_mass(self, residue: str) -> float:
        """Get the mass of a residue.

        Args:
            residue (str):
                The residue whose mass to fetch. This residue
                must be in the residue set or this will raise
                a `KeyError`.

        Returns:
            float: The mass of the residue in Daltons.
        """
        if self.residue_remapping and residue in self.residue_remapping:
            residue = self.residue_remapping[residue]
        return self.residue_masses[residue]

    def get_sequence_mass(self, sequence: str, charge: int | None) -> float:
        """Get the mass of a residue sequence.

        Args:
            sequence (str):
                The residue sequence whose mass to calculate.
                All residues must be in the residue set or
                this will raise a `KeyError`.
            charge (int | None, optional):
                Charge of the sequence to calculate the mass.

        Returns:
            float: The mass of the residue in Daltons.
                   If a charge is specified, returns m/z.
        """
        mass = sum([self.get_mass(residue) for residue in sequence]) + H2O_MASS
        if charge:
            mass = (mass / charge) + PROTON_MASS_AMU
        return float(mass)

    def tokenize(self, sequence: str | list[str] | None) -> list[str]:
        """Split a peptide represented as a string into a list of residues.

        Args:
            sequence (str | list[str] | None): The peptide to be split.

        Returns:
            list[str]: The sequence of residues forming the peptide.
        """
        if sequence is None:
            return []
        if isinstance(sequence, list):
            return sequence

        tokens = [
            item
            for sublist in re.findall(self.tokenizer_regex, sequence)
            for item in sublist
            if item
        ]

        # Filter out standalone modifications (capture group 1: brackets, parens, numbers)
        # Keep only amino acids (capture group 2: A-Z optionally with modifications)
        # Standalone mods cannot be looked up in residue_masses and cause KeyError in _novor_match
        filtered_tokens = []
        for token in tokens:
            # Amino acids always start with uppercase letter A-Z
            if token and token[0].isalpha() and token[0].isupper():
                filtered_tokens.append(token)

        return filtered_tokens

    def detokenize(self, sequence: list[str]) -> str:
        """Joining a list of residues into a string representing the peptide.

        Args:
            sequence (list[str]):
                The sequence of residues.

        Returns:
            str:
                The string representing the peptide.
        """
        return "".join(sequence)

    def _strip_ptm(self, residue: str) -> str | None:
        """Strip PTM modifications from a residue token.

        Handles common PTM formats:
        - C+57.021 -> C
        - M+15.995 -> M
        - N+0.984 -> N
        - Q+0.984 -> Q
        - C[UNIMOD:4] -> C
        - M(ox) -> M
        - -17.027 -> None (standalone N-terminal modifications are skipped)
        - +42.01 -> None (standalone modifications are skipped)
        - etc.

        Args:
            residue: A residue token possibly with PTM modification.

        Returns:
            The base amino acid character without modifications, or None if
            the token is a standalone modification that should be skipped.
        """
        if not residue or len(residue) == 0:
            return residue

        # If it's already a known residue, return as-is
        if residue in self.residue_to_index or residue in self.residue_remapping:
            return residue

        # Check for standalone modifications (e.g., -17.027, +42.01, (Acetyl), [UNIMOD:1])
        # These are N-terminal or C-terminal modifications without an amino acid
        # They should be skipped during encoding
        first_char = residue[0] if len(residue) > 0 else ''
        if first_char in '+-.' or first_char.isdigit() or first_char == '(' or first_char == '[':
            # This is a standalone modification, skip it
            return None

        # Extract the first uppercase letter as the base amino acid
        # This handles: C+57.021, M+15.995, N+0.984, C[UNIMOD:4], M(ox), etc.
        if len(residue) >= 1 and residue[0].isupper() and residue[0].isalpha():
            base_aa = residue[0]
            # Check if base amino acid is valid
            if base_aa in self.residue_to_index or base_aa in self.residue_remapping:
                return base_aa
            # Map I to L for consistency (common in proteomics) if I is not in vocab
            if base_aa == 'I' and 'L' in self.residue_to_index:
                return 'L'

        # If we can't parse it, return None to skip this unknown residue
        # This handles rare amino acids like 'O' (Pyrrolysine) and 'U' (Selenocysteine)
        # that are not in the standard vocabulary
        return None

    def encode(
        self,
        sequence: list[str],
        add_eos: bool = False,
        return_tensor: str | None = None,
        pad_length: int | None = None,
        strip_ptm: bool = True,
    ) -> torch.LongTensor | np.ndarray:
        """Map a sequence of residues to their indices and optionally pad them to a fixed length.

        Args:
            sequence (list[str]):
                The sequence of residues.
            add_eos (bool):
                Add an EOS token when encoding.
                Defaults to `False`.
            return_tensor (str | None, optional):
                Return type of encoded tensor. Returns a list if integers
                if no return type is specified. Options: None, pt, np
            pad_length (int | None, optional):
                An optional fixed length to pad the encoded sequence to.
                If this is `None`, no padding is done.
            strip_ptm (bool):
                If True, automatically strip PTM modifications from unknown residue tokens.
                This handles formats like C+57.021, M+15.995, N+0.984, C[UNIMOD:4], M(ox).
                Defaults to `True`.

        Returns:
            torch.LongTensor | np.ndarray:
                A tensor with the indices of the residues.
        """
        encoded_list = []
        for residue in sequence:
            # First try remapping
            if residue in self.residue_remapping:
                residue = self.residue_remapping[residue]
            # If still not found and strip_ptm is enabled, try stripping PTM
            elif strip_ptm and residue not in self.residue_to_index:
                residue = self._strip_ptm(residue)
                # Skip if _strip_ptm returned None (standalone modification)
                if residue is None:
                    continue
                # Apply remapping again after stripping
                if residue in self.residue_remapping:
                    residue = self.residue_remapping[residue]

            encoded_list.append(self.residue_to_index[residue])

        if add_eos:
            encoded_list.extend([self.EOS_INDEX])

        if pad_length:
            encoded_list.extend((pad_length - len(encoded_list)) * [self.PAD_INDEX])

        if return_tensor == "pt":
            return torch.tensor(encoded_list, dtype=torch.long)
        elif return_tensor == "np":
            return np.array(encoded_list, dtype=np.int32)
        else:
            return encoded_list

    def decode(self, sequence: torch.LongTensor | list[int], reverse: bool = False) -> list[str]:
        """Map a sequence of indices to the corresponding sequence of residues.

        Args:
            sequence (torch.LongTensor | list[int]):
                The sequence of residue indices.
            reverse (bool):
                Optionally reverse the decoded sequence.

        Returns:
            list[str]:
                The corresponding sequence of residue strings.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        residue_sequence = []
        for index in sequence:
            if index == self.EOS_INDEX:
                break
            if index == self.SOS_INDEX or index == self.PAD_INDEX:
                continue
            residue_sequence.append(index)

        if reverse:
            residue_sequence = residue_sequence[::-1]

        return [self.index_to_residue[index] for index in residue_sequence]

    def __len__(self) -> int:
        return len(self.index_to_residue)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueSet):
            return NotImplemented
        return self.vocab == other.vocab
