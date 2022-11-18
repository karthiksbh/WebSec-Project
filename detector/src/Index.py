# Importing libraries
import joblib
import detector.src.Inputscript as Inputscript

def main(url):
    # Load the pickle file
    classifier = joblib.load("detector/src/final_model/rf_final.pkl")
    
    # Checking and Predicting
    predictionDict = Inputscript.main(url)
    checkprediction = []

    for key in predictionDict:
        checkprediction.append(predictionDict[key])

    prediction = classifier.predict([checkprediction])
    
    if(prediction[0] == 1):
        print("PHISHING site")
    elif(prediction[0] == -1):
        print("LEGITIMATE site")
    else:
        print("SUSPICIOUS site")

    return(prediction[0])