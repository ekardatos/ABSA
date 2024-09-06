# Import library for parsing and manipulating xml files
import xml.etree.ElementTree as ET

# Read the dataset
file = ET.parse('C:/Users/Ευάγγελος Καρδάτος/Desktop/Final Text/ABSA16_Restaurants_Train_SB1_v2.xml')

# Find the instances of 'Review' with clild
reviews = file.findall('Review')

# Create a list for individual review
my_list = []

# For every single review in 'reviews'
for rev in reviews:
    # Review to string
    my_list.append(ET.tostring(rev))

# Number of files we want in every file
revs_in_file = 35

for review in range(0, len(my_list), revs_in_file):
    with open('C:/Users/Ευάγγελος Καρδάτος/Desktop/Final Text/Xml Files/part' + str((review//revs_in_file)+1)+".xml","wb") as f:
        # Place the header on top of the file
        f.write(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        # Firts tag <Reviews> element for every review
        f.write(b"<Reviews>\n")
        # Change the row and append every single of 35 files (36-1) in the file.
        f.write(b"\n".join(my_list[x] for x in range(review, min(review + revs_in_file, len(my_list) - 1))))
        # Last tag <Reviews> element for every review
        f.write(b"\n </Reviews>")
