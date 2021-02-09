import csv 
  
# csv file name 
filename = "/content/SarcasmDetection/resource/test/narendramodi_final.csv"
file = open('/content/SarcasmDetection/resource/text_data.txt',"w")
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = next(csvreader) 
    for row in csvreader: 
        l = list(row[1].strip())
        label='0' 
        pre_list = ['T', 'e', 's', 't','\t',label,'\t']
        pre_list+= l
        text=""
        for i in pre_list:
           text+=i 
        #print(text)  
        file.writelines(text + '\n')
