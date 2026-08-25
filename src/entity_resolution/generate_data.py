"""Data generation module for creating synthetic Canadian person records."""

import random
from dataclasses import dataclass, asdict
from typing import List, Optional
from faker import Faker
from faker.providers import BaseProvider


class CanadianAddressProvider(BaseProvider):
    """Custom provider for Canadian addresses."""
    
    CANADIAN_PROVINCES = [
        ('AB', 'Alberta'), ('BC', 'British Columbia'), ('MB', 'Manitoba'),
        ('NB', 'New Brunswick'), ('NL', 'Newfoundland and Labrador'),
        ('NS', 'Nova Scotia'), ('NT', 'Northwest Territories'),
        ('NU', 'Nunavut'), ('ON', 'Ontario'), ('PE', 'Prince Edward Island'),
        ('QC', 'Quebec'), ('SK', 'Saskatchewan'), ('YT', 'Yukon')
    ]

    # Resident population by province/territory from the 2021 Census of
    # Population (Statistics Canada), used to weight province selection so the
    # synthetic population mirrors the real provincial distribution.
    POPULATION_2021 = {
        'AB': 4_262_635, 'BC': 5_000_879, 'MB': 1_342_153, 'NB': 775_610,
        'NL': 510_550, 'NS': 969_383, 'NT': 41_070, 'NU': 36_858,
        'ON': 14_223_942, 'PE': 154_331, 'QC': 8_604_495, 'SK': 1_132_505,
        'YT': 40_232,
    }
    
    # Canada Post forward-sortation-area prefix letters, by province/territory.
    # The first letter of a Canadian postal code is determined entirely by the
    # province or territory (NT and NU share the X prefix).
    POSTAL_PREFIX_BY_PROVINCE = {
        'NL': ['A'],
        'NS': ['B'],
        'PE': ['C'],
        'NB': ['E'],
        'QC': ['G', 'H', 'J'],
        'ON': ['K', 'L', 'M', 'N', 'P'],
        'MB': ['R'],
        'SK': ['S'],
        'AB': ['T'],
        'BC': ['V'],
        'NT': ['X'],
        'NU': ['X'],
        'YT': ['Y'],
    }
    
    STREET_TYPES = ['Street', 'Avenue', 'Road', 'Boulevard', 'Drive', 'Lane', 'Court', 'Place', 'Way', 'Crescent']
    
    def canadian_province(self) -> tuple:
        """Return a province/territory weighted by its 2021 Census population."""
        weights = [self.POPULATION_2021[code] for code, _ in self.CANADIAN_PROVINCES]
        return random.choices(self.CANADIAN_PROVINCES, weights=weights, k=1)[0]
    
    def canadian_postal_code(self, prov_code: Optional[str] = None) -> str:
        """Generate a valid Canadian postal code (A1A 1A1) whose FSA prefix
        letter is consistent with the given province/territory code.

        When ``prov_code`` is omitted a province is chosen uniformly at random,
        so the returned code is always valid for *some* province. The prefix is
        drawn from :attr:`POSTAL_PREFIX_BY_PROVINCE`, enforcing the real-world
        correspondence between region and prefix letter.
        """
        if prov_code is None:
            prov_code = self.canadian_province()[0]
        prefix_letters = self.POSTAL_PREFIX_BY_PROVINCE[prov_code]
        valid = 'ABCEGHJKLMNPRSTVXY'
        numbers = '0123456789'
        return (f"{random.choice(prefix_letters)}{random.choice(numbers)}{random.choice(valid)} "
                f"{random.choice(numbers)}{random.choice(valid)}{random.choice(numbers)}")
    
    def canadian_address(self) -> str:
        street_num = random.randint(1, 9999)
        street_name = self.generator.street_name()
        street_type = random.choice(self.STREET_TYPES)
        city = self.generator.city()
        prov_code, prov_name = self.canadian_province()
        postal = self.canadian_postal_code(prov_code)
        return f"{street_num} {street_name} {street_type}, {city}, {prov_code} {postal}"


@dataclass
class Person:
    """Person record with Canadian address."""
    first_name: str
    last_name: str
    date_of_birth: str  # YYYY-MM-DD format
    address: Optional[str] = None
    email: Optional[str] = None
    
    def to_text(self) -> str:
        """Convert to text for embedding."""
        parts = [
            f"First Name: {self.first_name}",
            f"Last Name: {self.last_name}",
            f"Date of Birth: {self.date_of_birth}",
        ]
        if self.address:
            parts.append(f"Address: {self.address}")
        if self.email:
            parts.append(f"Email: {self.email}")
        return "\n".join(parts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Splink."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Person':
        return cls(**data)


# Share of the Canadian population that is male (Statistics Canada, 2021 Census:
# 49.3% male). First names are drawn gender-consistently with this probability so
# the synthetic population is not gender-imbalanced relative to the real one.
MALE_FRACTION = 0.493


def generate_person(fake: Faker, missing_rate: float = 0.3) -> Person:
    """Generate a single person with optional missing fields."""
    # Gender-correlated given names: ~49.3% of records are male, and each
    # person's first name is drawn from the gender-appropriate name list so the
    # male/female name mix stays close to the real 49.3/50.7 split.
    is_male = random.random() < MALE_FRACTION
    first_name = fake.first_name_male() if is_male else fake.first_name_female()
    last_name = fake.last_name()
    # Generate DOB between 1930 and 2010
    dob = fake.date_of_birth(minimum_age=14, maximum_age=94).strftime('%Y-%m-%d')
    
    # 30% chance of missing address and/or email
    address = None
    email = None
    
    if random.random() > missing_rate:
        address = fake.canadian_address()
    
    if random.random() > missing_rate:
        # Realistic, varied email addresses (Faker internet provider).
        email = fake.ascii_safe_email()
    
    return Person(
        first_name=first_name,
        last_name=last_name,
        date_of_birth=dob,
        address=address,
        email=email
    )


def generate_people(count: int = 50000, missing_rate: float = 0.3, seed: int = 42) -> List[Person]:
    """Generate a list of people with configurable missing data rate."""
    random.seed(seed)
    Faker.seed(seed)  # also seed Faker's RNG so generation is reproducible
    fake = Faker('en_CA')
    fake.add_provider(CanadianAddressProvider)
    
    people = []
    for _ in range(count):
        people.append(generate_person(fake, missing_rate))
    
    return people


def introduce_variations(person: Person, variation_rate: float = 0.1) -> Person:
    """Create a slightly modified version of a person (for testing matches)."""
    fake = Faker('en_CA')
    fake.add_provider(CanadianAddressProvider)
    
    new_person = Person(
        first_name=person.first_name,
        last_name=person.last_name,
        date_of_birth=person.date_of_birth,
        address=person.address,
        email=person.email
    )
    
    # Introduce typos/variations
    if random.random() < variation_rate:
        # Typo in first name
        name = list(new_person.first_name)
        if len(name) > 2:
            idx = random.randint(0, len(name) - 1)
            name[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        new_person.first_name = ''.join(name)
    
    if random.random() < variation_rate:
        # Typo in last name
        name = list(new_person.last_name)
        if len(name) > 2:
            idx = random.randint(0, len(name) - 1)
            name[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        new_person.last_name = ''.join(name)
    
    if random.random() < variation_rate and new_person.address:
        # Address variation (e.g., St -> Street, Ave -> Avenue)
        addr = new_person.address
        replacements = [
            ('St ', 'Street '), ('St.', 'Street'), ('Ave ', 'Avenue '),
            ('Ave.', 'Avenue'), ('Rd ', 'Road '), ('Rd.', 'Road'),
            ('Blvd ', 'Boulevard '), ('Blvd.', 'Boulevard')
        ]
        for old, new in replacements:
            if old in addr and random.random() < 0.5:
                addr = addr.replace(old, new)
        new_person.address = addr
    
    if random.random() < variation_rate and new_person.email:
        # Email variation
        email = new_person.email
        if random.random() < 0.5:
            # Add a number
            local, domain = email.split('@')
            email = f"{local}{random.randint(1, 99)}@{domain}"
        new_person.email = email
    
    return new_person


if __name__ == '__main__':
    import argparse
    from .vector_store import build_person_store
    from .model_pins import EMBEDDING_MODEL_ID

    parser = argparse.ArgumentParser(
        description="Generate people and persist their FAISS index."
    )
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL_ID,
    )
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    people = generate_people(args.count, args.missing_rate, args.seed)
    store = build_person_store(people, args.model)
    store.save(args.output_dir)
    print(f"Generated {len(people):,} people")
    print(f"Saved FAISS index and metadata to {args.output_dir}")