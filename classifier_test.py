from pickle import load as pickle_load

# load the model and vectorizer
print("loading model and vectorizer")
with open("model.pkl", "rb") as f:
    model = pickle_load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle_load(f)
print("loaded model and vectorizer")

# test the model with some data for ham
print("Ham testing")
ham_data = vectorizer.transform(
    [
        "Thank you, ABC. Can you also share your LinkedIn profile? As you are a good at programming at python, would be willing to see your personal/college projects.",
        "Hi yall, We have a Job Openings in the positions of software engineer, IT officer at ABC Company.Kindly, send us your resume and the cover letter as soon as possible if you think you are an eligible candidate and meet the criteria.",
        "Dear ABC, Congratulations! You have been selected as a SOftware Developer at XYZ Company. We were really happy to see your enthusiasm for this vision and mission. We are impressed with your background and we think you would make an excellent addition to the team.",
    ]
)
probabilities = model.predict_proba(ham_data)
predictions = model.predict(ham_data)
for i in range(ham_data.shape[0]):
    print(f"Spam probability: {probabilities[i][1]:.5f}, Type: {predictions[i]}")
print("done")

# test the model with some data for spam
print("Spam testing")
spam_data = vectorizer.transform(
    [
        "congratulations, you became today's lucky winner",
        "1-month unlimited calls offer Activate now",
        "Ram wants your phone number for sex calls",
    ]
)
probabilities = model.predict_proba(spam_data)
predictions = model.predict(spam_data)
for i in range(spam_data.shape[0]):
    print(f"Spam probability: {probabilities[i][1]:.5f}, Type: {predictions[i]}")
print("done")
