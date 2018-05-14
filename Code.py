
# coding: utf-8

# Ich habe vor kurzem ein Buch gelesen, in dem die Stimmung im Verlaufe des Buches nach unten ging. Jetzt habe ich mich gefragt: Kann man das irgendwie automatisiert nachvollziehen? 
# 
# Als Ansatz habe ich mir folgendes überlegt: Word2Vec-Modelle geben mir die Möglichkeit, Worte in einen n-dimensionalen Raum zu mappen, sodass Worte, die häufig im selben Kontext (bezogen auf den Korpus, auf dem trainiert wird) erscheinen, nah aneinander sind. Synonyme haben also geringe Distanz, "Buch" und "lesen" haben geringe Distanz voneinander etc.
# 
# Google stellt ein solches vortrainiertes Modell bereit, dass auf Google-News-Artikeln trainiert wurde und auf 300 Dimensionen mappt. Ich werde dieses benutzen und dann mal schauen, was sich ergibt.
# 
# Das Buch habe ich als HTML-Datei lokal auf dem Rechner. 

# In[1]:


file = open('Best_Served_Cold.html', "r") 
text=file.read().lower()


# In[2]:


text[:1000]


# In[3]:


import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display, HTML


pd.options.display.max_colwidth = 500 # Damit man beim Anzeigen von Text in DataFrames den ganzen Satz lesen kann


# Zuerst muss ich mich um die html-Tags kümmern. Dazu lasse ich mir alle anzeigen und entscheide dann, ob ich die nicht einfach alle entfernen kann.

# In[4]:


listOfTags = list()
textToSearch = text
while True:
    positionStart = textToSearch.find("<")
    positionEnd = textToSearch.find(">") + 1
    listOfTags.append(textToSearch[positionStart:positionEnd])
    textToSearch = textToSearch[positionEnd:]
    if positionStart == -1:  # no more Tags
        break
        
pd.Series(listOfTags).value_counts()[:10]


# Das kann alles weg.

# In[5]:


#delete Tags
text2=text
while True:
    positionStart=text2.find("<")
    positionEnd=text2.find(">") +1
    if positionStart==-1 or positionEnd==-1 : #no more Tags
      break
    text2=text2[:positionStart] +  text2[positionEnd:]


#replace line breaks/nbsp
text3=text2     .replace("\n"," ")     .replace("&nbsp;"," ")     .replace("&nbsp"," ") 
#multiple whitespaces to 1
text3=re.sub("\s+"," ",text3)
text3[:1000]


# Als nächstes kümmere ich mich um das Trennen von Sätzen. Dafür ist für mich interessant, ob ich Punkte nehmen kann oder ob häufig ein Begriff abgekürzt wird.

# In[6]:


#a word before . 
pattern="[A-z]+\." 
matches = re.findall(pattern,text3)
pd.Series(matches).value_counts()[:20]


# Nein, sieht nicht so aus. Ich ersetze jetzt noch ein paar Dinge (andere Satzzeichen zu Punkten, Abkürzungen zum vollen Äquivalent) und entferne Sonderzeichen.
# 
# Dann kann ich den Text als Liste von Sätzen darstellen, die jeweils Listen von Wörtern sind.

# In[7]:


text4=text3         .replace("?",".")        .replace("!",".")        .replace("’","'")         .replace("ain't","is not")         .replace("can't","cannot")         .replace("'ll"," will")         .replace("n't"," not")          .replace("'m"," am")            .replace("'d", " would")         .replace("'ve", " have")         .replace("'re", " are")          .replace("'em", " them")


# In[8]:


marks=[",",";","(",")",'"','”','“',"",":","...","…","'s","'"]
for mark in marks:
    text4=text4.replace(mark,"")
text4=text4.replace("-"," ")


# In[9]:


text5=text4.split(".")
text6=[sentence.split(' ') for sentence in text5]
text6[:2]


# Als nächstes werde ich die Worte bezüglich ihrer Ähnlichkeit zu *evil* scoren. Dafür habe ich mir ein pre-trainend word2vec-Model, das von Google bereit gestellt wird, heruntergeladen.

# In[10]:


import gensim

path=r'''GoogleNews-vectors-negative300.bin'''
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)


# In[11]:


wordToCompare="evil"

l2=list()
listWordsNotFound=list()
for sentence in text6:
    sentenceList=list()
    for word in sentence:
        try:
            sim=model.similarity(word,wordToCompare)
            sentenceList.append((word,sim))
        except:
           listWordsNotFound.append(word)
    l2.append(sentenceList)


# In[12]:


l2[:20]


# In[13]:


pd.Series(listWordsNotFound).value_counts()[:30]


# Hauptsächlich Eigennamen. Mir nicht ganz klar, warum er Axt nicht kennt. Das mit Armour und grey ist schade, wahrscheinlich kennt er dann auch colour, neighbour etc. nicht. Das wäre mir jetzt aber zu viel Aufwand, das zu ersetzen.
# 
# Es fällt mir auch auf, dass *Shivers* fehlt. Das ist hier ein Eigenname, wird aber wahrscheinlich wirklich als Zittern erkannt. Man könnte vor dem toLower() das noch ersetzen, aber ich denke, das wird auch so passen.
# 
# 
# Jetzt habe ich jedes Wort auf die Ähnlichkeit seines (durchschnittlichen) Kontextes zum Kontext von *evil* gemappt. Jetzt ist noch die Frage, wie ich das am besten aggregiere. Mit "einfach pro Satz mitteln" scheint man wohl gute Erfahrungen gemacht zu haben.

# In[14]:


def score(list_sentence, func_aggregate_sentences):
    list_sentence_scored=[func_aggregate_sentences(sentence) for sentence in list_sentence]   
    
    #Verlauf der Scores anzeigen
    plt.figure(figsize=(25,10))
    values=[tupel[1] for tupel in list_sentence_scored]
    plt.plot(values)

    #Top 20 Sätze bezüglich aggregierter Similarity anzeigen
    display(pd.DataFrame(list_sentence_scored,columns=["text","value"])         .sort_values(by=["value"], ascending=False)          [:20]
    )
    



def agg_sentence_mean(sentence):
    if len(sentence) == 0:
        return ("",float("NaN"))
    else:
        text=" ".join([tupel[0] for tupel in sentence])  
        values=[tupel[1] for tupel in sentence]
        mean=sum(values)/len(values)
        return (text,mean)
    
score(l2,agg_sentence_mean)


# Man sieht jetzt keine wirkliche Entwicklung, aber man sieht Spitzen - das ist sehr interessant!
# 
# Grundsätzlich scheint es also zu funktionieren, aber es sticht ins Auge, dass es meistens 1-Wort-Sätze/Fragen/Ausrufe sind. Scheinbar werden die wichtigen Wörte in langen Sätzen zu sehr verdünnt.
# 
# Ich sehe da spontan 2 sinnvolle Abhilfen:
# * beachte pro Satz die maximale Ähnlichkeit
# * berechne einen gewichteten Durchschnitt. Hier fällt mir spontan TDIF ein, um vernünftige Gewichte zu finden.
# 
# Der 1. Ansatz ist einfacher, also fange ich mal damit an.

# In[15]:


def agg_sentence_max(sentence):
    
    if len(sentence) == 0:
        return ("",float("NaN"))
    else:
        text=" ".join([tupel[0] for tupel in sentence])  
        max_value=max([tupel[1] for tupel in sentence])
        return (text,max_value)

score(l2,agg_sentence_max)


# Ich glaube, das war nicht sinnvoll. Im Mittelfeld hat sich gefühlt nichts bewegt und es werden jetzt einfach die Sätze mit "Evil" angezeigt...
# 
# Ich schau mal, was passiert, wenn ich "evil" einfach rausnehme.

# In[16]:


l3=[[word for word in sentence if word[0]  != "evil"] for sentence in l2]

score(l3,agg_sentence_max)


# Ja, damit hätte man rechnen müssen. Es werden jetzt bestimmte Worte gefunden (supernatural, sinister, murderous) und diese Sätze kommen dann noch oben. 
# 
# Spontan fallen mir noch 2 weitere Aggregationen ein:
# 
# * berechne den Durchschnitt der Top k, sagen wir 3 Wörter
# * berechne eine gewichtete Summe: Max+1/2 &ast; Second+1/3 &ast; Third usw. usf., dann noch bezüglich der Länge normalisieren

# In[17]:



#Das Austesten mehrere k ist bequemer mit Currying: Abhänging von k gebe ich hier eine unterschiedliche Funktion zurück
def agg_sentence_k_highest(k):
    def agg_sentence(sentence):
        if len(sentence) == 0:
            return ("",float("NaN"))
        else:
            text=" ".join([tupel[0] for tupel in sentence])  
            top_k=sorted([tupel[1] for tupel in sentence])[-k:]
            top_k_mean=sum(top_k)/len(top_k)
            return (text,top_k_mean)
    return agg_sentence


# In[18]:


score(l3,agg_sentence_k_highest(3))


# Hier sieht man jetzt wieder, dass ich in dem Moment, wo ich den Durchschnitt betrachte, solche 1-Wort-Sätze ganz stark dabei sind. Ich probiere es nochmal mit k=5, aber perspektivisch wird das eher noch schlimmer sein.
# 
# Aber mir gefällt schon, was ich an den Sätzen sehe. Die Entwicklung ist immer noch ... nicht sichtbar.

# In[19]:


score(l3,agg_sentence_k_highest(5))


# Jup, noch schlimmer.
# 
# Der nächste Ansatz war über eine gewichtete Summe: Ich sortiere nach der Ähnlichkeit, dann bekommt das erste Wort Faktor 1, das 2. Wort Faktor 1/2, das i. Wort Faktor 2^(-i+1). Die maximale Summa an Faktoren für n Wörter ist dann  2-2^(-n+1).
# 

# In[20]:


def agg_sentence_weighted_sum(sentence):
    if len(sentence) == 0:
        return ("",float("NaN"))
    else:
        text=" ".join([tupel[0] for tupel in sentence])  
        
        #man beachte, dass hier jetzt 2^-i, weil das 1. Wort im Satz das Wort 0 in der Liste ist
        values_sorted=sorted([tupel[1] for tupel in sentence],reverse=True)
        weighted_values=[2**(-i)*value for (i,value) in enumerate(values_sorted)] 
        normalization_factor=2-2**(-len(weighted_values)+1)
                
        top_k_mean=sum(weighted_values)/normalization_factor
        return (text,top_k_mean)


# In[21]:


score(l3,agg_sentence_weighted_sum)


# Es gefällt mir schon nicht schlecht. Mal sehen, was passiert, wenn ich einfach nur Sätze mit mindestens 3 Worten betrachte

# In[22]:


l4=[sentence for sentence in l3 if len(sentence)>=3]
score(l4,agg_sentence_weighted_sum)


# Um eine Entwicklung zu beobachten hilft das nicht so gut, aber mir gefällt die Auswahl der Sätze **sehr**. Ich gieße das ganze mal in eine Methode und spiele damit etwas rum.

# In[23]:


def get_sentences(text, word_to_compare, number_words_output):
    

    
    list_sentences=list()
    list_words_not_found=list()
    print(f"Berechne Similarites",end="\r")
    for i,sentence in enumerate(text):

        result_sentence=list()
        for word in sentence:
            try:
                if word != word_to_compare :
                    sim=model.similarity(word,word_to_compare)
                    result_sentence.append((word,sim))
            except:
               list_words_not_found.append(word)
        list_sentences.append(result_sentence)

    list_sentences=[sentence for sentence in list_sentences if len(sentence)>=3]
 
    def agg_sentence_weighted_sum(sentence):
        if len(sentence) == 0:
            return ("",float("NaN"))
        else:
            text=" ".join([tupel[0] for tupel in sentence])  

            #man beachte, dass hier jetzt 2^-i, weil das 1. Wort im Satz das Wort 0 in der Liste ist
            values_sorted=sorted([tupel[1] for tupel in sentence],reverse=True)
            weighted_values=[2**(-i)*value for (i,value) in enumerate(values_sorted)] 
            normalization_factor=2-2**(-len(weighted_values)+1)

            top_k_mean=sum(weighted_values)/normalization_factor
            return (text,top_k_mean)


    list_scored_sentences=[agg_sentence_weighted_sum(sentence) for sentence in list_sentences]  

    display(pd.DataFrame(list_scored_sentences,columns=["text","value"])         .sort_values(by=["value"], ascending=False)          [:number_words_output]
    )


# In[24]:


get_sentences(text6,"love",10)


# In[25]:


get_sentences(text6,"city",10)


# Bei "Love" sieht man schon Probleme, dass Wort-Varianten wie "loved" nach oben gepusht werden.
# Bei "City" dominiert das Quasi-Synonym "Town", aber Satz 7 und 10 gefallen mir sehr!
# 
# Das mit Wort-Varianten kann man mit Stemming beheben. Für das mit Synonymen *kann* man überlegen, ob man eventuell vorher einen Thesaurus drüber laufen lässt und alle Synonyme des Wortes auf das Wort mappt - oder der Methode eine Blackliste übergeben kann, wo ich z.B. hier "cities" und "town" mitgeben würde.
# 
# Insgesamt wäre es auch schöner, wenn man nicht den gefilterten Text angezeigt bekommt, sondern den "rohen" Text.

#  

# 
# 
# 
# 
# 
# To be continued...
