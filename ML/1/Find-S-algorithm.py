import csv
import streamlit as st

def find_s_algorithm():
    hypo = ['%','%','%','%','%','%']

    with open('trainingdata.csv') as csv_file:
        readcsv = csv.reader(csv_file, delimiter=',')
        st.write(readcsv)

        data = []
        st.write("\nThe given training examples are:")
        for row in readcsv:
            st.write(row)
            if row[len(row)-1].upper() == "YES":
                data.append(row)

    st.write("\nThe positive examples are:")
    for x in data:
        st.write(x)
    st.write("\n")

    TotalExamples = len(data)
    i=0
    j=0
    k=0
    st.write("The steps of the Find-s algorithm are:\n",hypo)
    list = []
    p=0
    d=len(data[p])-1
    for j in range(d):
        list.append(data[i][j])
    hypo=list
    i=1
    for i in range(TotalExamples):
        for k in range(d):
            if hypo[k]!=data[i][k]:
                hypo[k]='?'
                k=k+1        
            else:
                hypo[k]
        st.write(hypo)
    i=i+1

    st.write("\nThe maximally specific Find-s hypothesis for the given training examples is:")
    list=[]
    for i in range(d):
        list.append(hypo[i])
    st.write(list)

find_s_algorithm()