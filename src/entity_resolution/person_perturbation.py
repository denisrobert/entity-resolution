"""Clerical-error perturbation model for Person records.

The :class:`PersonPerturbator` takes a base :class:`~generate_data.Person` and
applies exactly *one* of six perturbation types (chosen explicitly or at
random). Typos model clerical errors and draw from four classic mechanisms:

* **addition** -- an extra character is typed;
* **deletion** -- a character is dropped ("subtractions");
* **substitution** -- the wrong character is typed;
* **metathesis** -- two adjacent characters are typed in swapped order.

The six perturbation types::

    INITIAL_FIRST_NAME   perturbation 1 -- keep only the first initial of the
                         first name (e.g. ``John`` -> ``J.``)
    TYPO_IDENTITY        perturbation 2 -- a typo in first name, last name, or
                         date of birth
    TYPO_ADDRESS         perturbation 3 -- a typo (or two) in the address
    DENORMALIZE_ADDRESS  perturbation 4 -- reorder address fields / abbreviate
                         street types (no typo: only the representation changes)
    TYPO_EMAIL           perturbation 5 -- a typo in the local part or a domain
                         label of the email
    MISSING_OPTIONAL     perturbation 6 -- remove either the address or the
                         email altogether

The perturbator never mutates its input: every operation returns a new
``Person`` via :func:`dataclasses.replace`. A seeded instance is fully
reproducible, so experiments can generate deterministic noisy duplicates.

Basic usage::

    from person_perturbation import PersonPerturbator, Perturbation

    perturber = PersonPerturbator(seed=42)

    noisy = perturber.perturb(person)                          # random kind
    init  = perturber.perturb(person, Perturbation.INITIAL_FIRST_NAME)
    noisy, kind = perturber.perturb_with_kind(person)          # track the kind
    changed, kind = perturber.perturb_different(person)        # guaranteed diff
"""

from __future__ import annotations

import enum
import random
from dataclasses import replace
from typing import Any, Optional, Sequence, Union

from .generate_data import Person

# --- character alphabets for clerical errors --------------------------------
_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
_EMAIL_LOCAL = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
_EMAIL_DOMAIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
_ADDRESS_CHARS = _LETTERS + _DIGITS


class Perturbation(str, enum.Enum):
    """The set of perturbation types :class:`PersonPerturbator` can apply."""

    INITIAL_FIRST_NAME = "initial_first_name"
    TYPO_IDENTITY = "typo_identity"
    TYPO_ADDRESS = "typo_address"
    DENORMALIZE_ADDRESS = "denormalize_address"
    TYPO_EMAIL = "typo_email"
    MISSING_OPTIONAL = "missing_optional"


class PersonPerturbator:
    """Apply exactly one clerical perturbation to a :class:`Person`.

    Parameters
    ----------
    seed:
        Optional RNG seed for reproducible draws.
    rng:
        Optional pre-created :class:`random.Random` (takes precedence over
        ``seed``).
    """

    # Full street-type token -> its common abbreviated form, applied by
    # :meth:`denormalize_address`.
    STREET_TYPE_ABBREVIATIONS: dict[str, str] = {
        "Street": "St",
        "Avenue": "Ave",
        "Road": "Rd",
        "Boulevard": "Blvd",
        "Drive": "Dr",
        "Lane": "Ln",
        "Court": "Ct",
        "Place": "Pl",
        "Way": "Wy",
        "Crescent": "Cres",
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._rng = rng if rng is not None else random.Random(seed)

    # ------------------------------------------------------------------ utils

    @property
    def kinds(self) -> list[Perturbation]:
        """All perturbation kinds, in enum order."""
        return list(Perturbation)

    def choose_kind(self) -> Perturbation:
        """Pick a perturbation type uniformly at random."""
        return self._rng.choice(self.kinds)

    # ------------------------------------------------------------- public API

    def perturb(
        self,
        person: Person,
        kind: Optional[Union[Perturbation, str]] = None,
    ) -> Person:
        """Return a perturbed copy of ``person``.

        ``kind`` may be a :class:`Perturbation` member, one of its ``.value``
        strings, or ``None`` to choose one at random. When the perturbation
        targets an absent field (e.g. an email typo on a person with no email)
        the call returns a copy of the input unchanged.
        """
        kind = Perturbation(kind) if kind is not None else self.choose_kind()
        return self._dispatch(person, kind)

    def perturb_with_kind(
        self,
        person: Person,
        kind: Optional[Union[Perturbation, str]] = None,
    ) -> tuple[Perturbation, Person]:
        """Like :meth:`perturb` but also returns ``(kind, person)``."""
        kind = Perturbation(kind) if kind is not None else self.choose_kind()
        return kind, self._dispatch(person, kind)

    def perturb_different(
        self,
        person: Person,
        kind: Optional[Union[Perturbation, str]] = None,
        max_attempts: int = 25,
    ) -> tuple[Perturbation, Person]:
        """A perturbed copy that is guaranteed to differ from the input.

        Rare degenerate draws (e.g. a substitution with the same char, or a
        perturbation on an all-absent record) can yield an identical copy; this
        helper retries until the record changed. Raises ``ValueError`` if
        ``max_attempts`` draws all returned identical copies.
        """
        kind = Perturbation(kind) if kind is not None else self.choose_kind()
        for _ in range(max_attempts):
            candidate = self._dispatch(person, kind)
            if candidate.to_dict() != person.to_dict():
                return kind, candidate
        raise ValueError(
            f"could not perturb person with kind={kind!r} into a different "
            f"record within {max_attempts} attempts"
        )

    # ------------------------------------------------------------ the six ops

    def initial_first_name(self, person: Person) -> Person:
        """Perturbation 1: keep only the first initial of the first name."""
        if not person.first_name:
            return replace(person)
        return replace(person, first_name=f"{person.first_name[0]}.")

    def typo_identity(self, person: Person) -> Person:
        """Perturbation 2: a typo in the first name, last name, or DOB.

        The affected field is chosen at random among those present; names use
        letter typos, dates use digit typos.
        """
        candidates: list[tuple[str, str, str]] = []
        if person.first_name:
            candidates.append(("first_name", person.first_name, _LETTERS))
        if person.last_name:
            candidates.append(("last_name", person.last_name, _LETTERS))
        if person.date_of_birth:
            candidates.append(("date_of_birth", person.date_of_birth, _DIGITS))
        if not candidates:
            return replace(person)
        field, value, alphabet = self._rng.choice(candidates)
        if field == "date_of_birth":
            new_value = self._typo_digits(value)
        else:
            new_value = self._clerical_typo(value, alphabet)
        if new_value == value:
            new_value = self._force_alter(value, alphabet)
        return replace(person, **{field: new_value})

    def typo_address(self, person: Person) -> Person:
        """Perturbation 3: a clerical typo (or two) in the address.

        A single typo is the norm; a second typo is applied with probability
        0.3. Typo spots favour house-number digits, then street or postal
        characters. The operation retries internally so an address typo always
        produces a change; only a ``None`` address (nothing to perturb) returns
        a copy unchanged.
        """
        if not person.address:
            return replace(person)
        address = person.address
        count = 2 if self._rng.random() < 0.3 else 1
        for attempt in range(32):
            current = address
            changed = False
            for _ in range(count):
                updated = self._typo_one_spot(current)
                if updated != current:
                    changed = True
                current = updated or current
            if changed:
                return replace(person, address=current)
        # Extremely unlucky: force a letter substitution in the street part.
        letter_positions = [i for i, ch in enumerate(address) if ch.isalpha()]
        if letter_positions:
            i = self._rng.choice(letter_positions)
            address = address[:i] + self._rng.choice(_LETTERS) + address[i + 1 :]
        return replace(person, address=address)

    def denormalize_address(self, person: Person) -> Person:
        """Perturbation 4: change how the address is written, no typo.

        One or two (picked at random) of: abbreviate the street-type token;
        reorder the comma-delimited fields (place the locality first); drop one
        comma separator.
        """
        if not person.address:
            return replace(person)
        parts = [p.strip() for p in person.address.split(",")]
        transforms = [
            self._abbreviate_street_type,
            self._reorder_fields,
            self._join_comma,
        ]
        picks = (
            [self._rng.choice(transforms)]
            if self._rng.random() < 0.5
            else self._rng.sample(transforms, 2)
        )
        out_parts = list(parts)
        for transform in picks:
            out_parts = transform(out_parts)
        return replace(person, address=", ".join(out_parts).strip())

    def typo_email(self, person: Person) -> Person:
        """Perturbation 5: a typo in the local part or a domain label."""
        if not person.email:
            return replace(person)
        email = self._typo_email_once(person.email)
        if email == person.email:
            email = self._force_typo_email(person.email)
        return replace(person, email=email)

    def missing_optional(self, person: Person) -> Person:
        """Perturbation 6: remove either the address or the email entirely."""
        if person.address and person.email:
            drop = "address" if self._rng.random() < 0.5 else "email"
            return replace(person, **{drop: None})
        if person.address:
            return replace(person, address=None)
        if person.email:
            return replace(person, email=None)
        return replace(person)

    # ------------------------------------------------------------- internals

    def _dispatch(self, person: Person, kind: Perturbation) -> Person:
        return {
            Perturbation.INITIAL_FIRST_NAME: self.initial_first_name,
            Perturbation.TYPO_IDENTITY: self.typo_identity,
            Perturbation.TYPO_ADDRESS: self.typo_address,
            Perturbation.DENORMALIZE_ADDRESS: self.denormalize_address,
            Perturbation.TYPO_EMAIL: self.typo_email,
            Perturbation.MISSING_OPTIONAL: self.missing_optional,
        }[kind](person)

    # --- clerical typo helpers ----------------------------------------------

    def _clerical_typo(
        self,
        text: str,
        alphabet: Sequence[str],
        positions: Optional[Sequence[int]] = None,
    ) -> str:
        """Apply one of addition / deletion / substitution / metathesis.

        ``positions`` restricts which index may change (used to keep date
        separators and the email ``@`` intact). Substitution draws a fresh
        character from ``alphabet``, insertion adds one, deletion removes one,
        and metathesis swaps two adjacent characters.
        """
        if not text:
            return text
        idxs = list(positions) if positions is not None else list(range(len(text)))
        if not idxs:
            return text
        op = self._rng.choice(["subtract", "insert", "delete", "transpose"])
        if op == "subtract":
            i = self._rng.choice(idxs)
            return text[:i] + text[i + 1 :]
        if op == "insert":
            i = self._rng.choice(idxs)
            return text[:i] + self._rng.choice(alphabet) + text[i:]
        if op == "transpose" and len(text) > 1:
            swaps = [i for i in idxs if i + 1 < len(text) and (i + 1) in idxs]
            if swaps:
                i = self._rng.choice(swaps)
                return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
            # fall through: substitution
            i = self._rng.choice(idxs)
            return text[:i] + self._rng.choice(alphabet) + text[i + 1 :]
        # substitution
        i = self._rng.choice(idxs)
        return text[:i] + self._rng.choice(alphabet) + text[i + 1 :]

    def _force_alter(self, text: str, alphabet: Sequence[str]) -> str:
        """Guarantee a changed string (retry until one sticks)."""
        if not text:
            return text
        idxs = list(range(len(text)))
        for _ in range(32):
            attempt = self._clerical_typo(text, alphabet, positions=idxs)
            if attempt != text:
                return attempt
        return text

    def _typo_digits(self, value: str) -> str:
        """A clerical error restricted to the digit positions of a date."""
        digit_positions = [i for i, ch in enumerate(value) if ch.isdigit()]
        if not digit_positions:
            return value
        return self._clerical_typo(value, _DIGITS, positions=digit_positions)

    def _typo_one_spot(self, address: str) -> str:
        """One typo in the address: house digits, then street/postal chars."""
        digit_positions = [i for i, ch in enumerate(address) if ch.isdigit()]
        if digit_positions and self._rng.random() < 0.55:
            i = self._rng.choice(digit_positions)
            return address[:i] + self._rng.choice(_DIGITS) + address[i + 1 :]
        letter_positions = [i for i, ch in enumerate(address) if ch.isalpha()]
        if letter_positions:
            i = self._rng.choice(letter_positions)
            return address[:i] + self._rng.choice(_LETTERS) + address[i + 1 :]
        return self._force_alter(address, _ADDRESS_CHARS)

    def _typo_email_once(self, email: str) -> str:
        """One typo, preferring the local part (60%) then one domain label."""
        local, sep, domain = email.partition("@")
        if not sep:
            return self._clerical_typo(local, _EMAIL_LOCAL)
        if local and self._rng.random() < 0.6:
            return self._clerical_typo(local, _EMAIL_LOCAL) + "@" + (domain or "")
        if domain:
            labels = domain.split(".")
            idx = self._rng.randrange(len(labels))
            label = labels[idx]
            altered = self._clerical_typo(label, _EMAIL_DOMAIN)
            if not altered:
                altered = self._force_alter(label, _EMAIL_DOMAIN) or "x"
            labels[idx] = altered
            return local + "@" + ".".join(labels)
        return local

    def _force_typo_email(self, email: str) -> str:
        """Guarantee a changed email (used when the first draw was identical)."""
        for _ in range(32):
            candidate = self._typo_email_once(email)
            if candidate != email:
                return candidate
        return email

    # --- address-denormalization helpers -------------------------------------

    def _abbreviate_street_type(self, parts: list[str]) -> list[str]:
        """Abbreviate the street-type token at the end of the street part."""
        if not parts:
            return parts
        first = parts[0]
        for full, abbr in self.STREET_TYPE_ABBREVIATIONS.items():
            if first == full or first.endswith(" " + full):
                prefix = first[: -len(full)].rstrip()
                parts[0] = f"{prefix} {abbr}" if prefix else abbr
                break
        return parts

    def _reorder_fields(self, parts: list[str]) -> list[str]:
        """Move all locality fields (everything after the street) in front."""
        if len(parts) <= 1:
            return parts
        street, rest = parts[0], parts[1:]
        return [*rest, street]

    def _join_comma(self, parts: list[str]) -> list[str]:
        """Remove one comma separator (street + locality merged)."""
        if len(parts) <= 1:
            return parts
        idx = self._rng.randrange(len(parts) - 1)
        if idx + 1 < len(parts):
            merged = f"{parts[idx]} {parts[idx + 1]}"
            parts = [*parts[:idx], merged, *parts[idx + 2 :]]
        return parts