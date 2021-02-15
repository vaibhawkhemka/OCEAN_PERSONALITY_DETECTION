import tweepy
import csv
from pandas import DataFrame
from googlesearch import search
import argparse

consumer_key = "JFuFyTOZn8TV6nmpz7rNYJOnh"
consumer_secret = "Colw3KW2vtFik0gN43sp9LHVkqBm9do55Q3TD4mhLOtZjNL1PF"
access_key = "1186964493667983360-DqW029snPabP9axNAdpbJJ4w5b1Yhy"
access_secret = "KdnhjmX9Fpp4cmZdlIwCd4O6pyDyG0p7OdsoNBzBKCAPC"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#x = "Chetan Bhagat, author, banker, iitian"

def id(x):
  query  = "twitter" + x
  for j in search(query, tld="com", num=1, stop=1, pause=2): 
    #print(j) 
    if(j.find("twitter.com")):
        atpos  = j.find('?')
        userID = j[20: atpos]
        #print(userID)
  return userID  

#print(userID)


def tw(userID,xf):
  tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended')
  outtweets = [[#tweet.id_str, 
              #tweet.created_at, 
              #tweet.favorite_count, 
              #tweet.retweet_count, 
              tweet.full_text.encode("utf-8").decode("utf-8")] 
             for idx,tweet in enumerate(tweets)]
  df = DataFrame(outtweets,columns=["text"]) #"id","created_at","favorite_count","retweet_count", 
  df.to_csv('/content/OCEAN_PERSONALITY_DETECTION/TEXT_FILES/TEXT/CSV_FILES/%s_tweets.csv' % xf,index=False)
def predict(inputfile,testcase):
   File = open(inputfile)
   for x in File:
     x = x.rstrip()
     at = x.find(",")
     xf = x[:at]
     print(xf)
     #print(id(xf))
     userID  = id(xf)
     #print(userID)
     tw(userID,xf)
if __name__=="__main__":
  parser = argparse.ArgumentParser(description='OCEAN PERSONALITY DETECTION')
  parser.add_argument(
    '--testcase',
    default=5,
    help='provide an integer (default: 5)')
  parser.add_argument(
    '--inputfile',
    default='/',
    help='provide location of text_file (default:)')
  arguments = parser.parse_args()
  inputfile = arguments.inputfile
  testcase  = arguments.testcase
  predict(inputfile,testcase)
  from subprocess import call
  call(["python", "test_handler.py"])

