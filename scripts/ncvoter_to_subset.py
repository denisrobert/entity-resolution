import csv
import os
import sys

DEFAULT_SOURCE = os.path.join("datasets", "ncvoter", "ncvoter_Statewide.csv")

def clean(value):
    return "" if value is None else value.strip()

def build_address(row):
    street = clean(row["res_street_address"])
    city = clean(row["res_city_desc"])
    state = clean(row["state_cd"])
    zip_code = clean(row["zip_code"])
    if not street:
        street = clean(row["mail_addr1"])
        if not city:
            city = clean(row["mail_city"])
        if not state:
            state = clean(row["mail_state"])
        if not zip_code:
            zip_code = clean(row["mail_zipcode"])
    parts = [p for p in [street, city, state, zip_code] if p]
    return ", ".join(parts)

def main(source=DEFAULT_SOURCE):
    base = os.path.splitext(source)[0]
    output = base + "_subset.csv"
    with open(source, "r", encoding="utf-8", errors="replace", newline="") as fin, \
         open(output, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.writer(fout)
        writer.writerow(["first_name", "last_name", "dob", "address", "email"])
        count = 0
        for row in reader:
            writer.writerow([
                clean(row["first_name"]),
                clean(row["last_name"]),
                clean(row["birth_year"]),
                build_address(row),
                "",
            ])
            count += 1
    print(f"Wrote {count} rows to {output}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE)
