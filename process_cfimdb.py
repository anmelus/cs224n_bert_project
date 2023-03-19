import csv

with open('data/ids-cfimdb-train.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    rows = []
    i = 0
    for row in reader:
        if (i == 0):
            i += 1
            rows.append(row)
        else:
            last_character = row[-1][-1] # get the last character in the row
            if last_character.isdigit(): # check if it's a digit
                multiplied_value = int(last_character) * 4 # multiply the last character by 5
                new_row = row[:-1] + [str(multiplied_value)] # replace the last column with the new value
                rows.append(new_row)

with open('data/cfimdb_scaled.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)