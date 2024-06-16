from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import mysql.connector
from mysql.connector import Error

# config.py
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Nivimonkey@123',
    'database': 'motordb',
    'raise_on_warnings': True,
}

def detail_execute_query(query):
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)  # Use dictionary cursor to fetch column names
            cursor.execute(query)
            
            # Fetch the first row
            result = cursor.fetchone()

            # Print the result in a neat format
            if result:
                print("Price:", result['prices'])
                print("Year:", result['motor_year'])
                print("Kms:", result['motor_km'])
                print("Seats:", result['motor_seats'])
                print("Power:", result['motor_powers'])
                print("Discount Offer:", result['discount_offer'])
                res = "Price:"+ str(result['prices'])+", Year:"+ str(result['motor_year'])+", Kms:"+ str(result['motor_km'])+", Seats:"+ str(result['motor_seats'])+", Motor Power:"+str(result['motor_powers'])+", Discount Offer:"+ str(result['discount_offer'])+", Location:"+ str(result['location'])+", city:"+ str(result['city'])+", Country:"+ str(result['country'])+", Area:"+ str(result['area'])
                return res
            else:
                print("No record found.")
                return "no record found, Please enter a valid vehicle name from the above mentioned list."
    except Error as e:
        print(f"Error: {e}")

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def execute_query(query):
    connection = None 
    try:
        connection = mysql.connector.connect(**DB_CONFIG)  
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result
             
    except Error as e:
        print(f"Error: {e}")

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def check_words_presence(sentence, word_list):
    count = 0
    for word in word_list:
        if word.lower() in sentence.lower():
            count += 1
            if count >= 2:
                return True
    return False

def check_words1_presence(sentence, word_list):
    count = 0
    for word in word_list:
        if word.lower() in sentence.lower():
            count += 1
            if count >= 1:
                return True
    return False

device = torch.device('cpu')

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chatbot")
async def chatbot(user_input: str = Form(...)):
    print(f"Received user input: {user_input}")

    car_list_check = ['buy','list','car','cars']
    bike_list_check = ['buy','list','bike','bikes']
    truck_list_check = ['buy','list','truck','trucks']
    number_of_seat = ['seat','seater']
    detail_check = ['detail','details','car','cars','bike','bikes','truck','trucks']


    if check_words_presence(user_input, truck_list_check):
        query = "SELECT title FROM motordb.mft_motors WHERE vehicle_type = 3;"
        res = execute_query(query)
        finstr = "Here are the list of available trucks present: \n"
        cnt = 1
        for row in res:
            finstr = finstr+str(cnt)+". "+row[0]+'                                           \n'
            cnt+=1
        finstr = finstr+ "                                          ---Please enter the corresponding vehicle name below to know more:-  #### follow_up:true , vehicle_type:3"
        return {"response": f"{finstr}"}
    
    elif check_words_presence(user_input, car_list_check):
        query = "SELECT title FROM motordb.mft_motors WHERE vehicle_type = 2;"
        res = execute_query(query)
        finstr = "Here are the list of available cars present: \n"
        cnt = 1
        for row in res:
            finstr = finstr+str(cnt)+". "+row[0]+'                                           \n'
            cnt+=1
        finstr = finstr+ "                                          ---Please enter the corresponding vehicle name below to know more:-  #### follow_up:true , vehicle_type:2"
        return {"response": f"{finstr}"}
    
    elif check_words_presence(user_input, bike_list_check):
        query = "SELECT title FROM motordb.mft_motors WHERE vehicle_type = 1;"
        res = execute_query(query)
        finstr = "Here are the list of available bikes present:                                           \n"
        cnt = 1
        for row in res:
            finstr = finstr+str(cnt)+". "+row[0]+' \n'
            cnt+=1
        finstr = finstr+ "                                          ---Please enter the corresponding vehicle name below to know more:-  #### follow_up:true , vehicle_type:1"
        return {"response": f"{finstr}"}
    
    else:
        with open('intents.json', 'r') as json_data:
            intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE, map_location=torch.device('cpu'))

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        bot_name = "Fredo Chatbot"

        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    response = response+ "#### follow_up:false , vehicle_type:0"
                    #data = {"data":response,"follow_up":False,"vehicle_type":None}
                    return {"response": f"{response}"}
        else:
            data = {'data':" I do not understand...",'follow-up':False}
            return {"response": f"{data}"}


@app.post("/vehicle_details")
async def chatbot(user_input: str = Form(...)):
    print(f"Received user input: {user_input}")
    query = f"SELECT location, country, city, area, prices, motor_year, motor_km, motor_seats, motor_powers, discount_offer FROM motordb.mft_motors WHERE title = '{user_input}';"
    try:
        res = detail_execute_query(query)
        finres = " Here is your requested details::- "+res 
        return {"response": f"{finres}"}
    except Exception as e:
        return {"reponse": "The car name you are searching for does not exist in our database .Please enter a valid car name."}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)