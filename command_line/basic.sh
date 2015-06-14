sed 's/,/;/' ./data/id_all_property.csv | sed 's/,/;/; s/(//g; s/)//g; s/{//g; s/}//g; s/,[0-9][0-9][0-9],/,/g ; s/,[0-9][0-9],/,/g;  s/,[0-9],/,/g; s/,[0-9]//;  s/www.www.auctionzip.ca/property_999999/ ; s/property/prp/g;  '  > ./data/id_all_property_no_count.csv

cat ./data/id_all_ip.csv | sed 's/,/;/ ; s/,/;/; s/{//g; s/}//g; ' > ./data/id_all_ip_v2.csv