import pickle

with open("db.model","rb") as f:
	model = pickle.load(f)

fs = float(input("enter cholestrol level "))
tr = float(input("enter testing blood pressure "))
th = float(input("enter maximum  heart rate achieved "))
fu = int(input(" 0 No and 1 Yes -->> fasting blood pressure "))
d = [[fs,tr,th,fu]]
res = model.predict(d)
print(res)


#data[["chol","trestbps","thalach","fbs"]]