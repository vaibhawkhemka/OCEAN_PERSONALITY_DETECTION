import csv 
 
# csv file name 
input_file = "names.txt"
Final_File = open('/content/OCEAN_PERSONALITY_DETECTION/TEXT_FILES/TEXT/information.txt',"w")
Input_File = open(input_file)
for x in Input_File:
     x = x.rstrip()
     at = x.find(",")
     xf = x[:at]
     CSV_File = "/content/OCEAN_PERSONALITY_DETECTION/TEXT_FILES/TEXT/CSV_FILES/"+xf+"_tweets.csv"

     with open(CSV_File, 'r') as csvfile: 
     # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
    
     # extracting field names through first row 
        fields = next(csvreader) 
        for row in csvreader: 
            l = list(row[0].strip())
            label='0' 
            #print(userID)
            pre_list = ['\t']
            pre_list+= l
            #print(pre_list)
            text=""
            for i in pre_list:
               if i!='\n':
                  text+=i 
            text= xf + text   
            #print(text)  
            Final_File.writelines(text + '\n')
        Final_File.writelines('\n')      
Final_File.close()               
from subprocess import call
call(["python", "OCEAN_MODEL.py"])
