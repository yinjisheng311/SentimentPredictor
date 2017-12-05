
# coding: utf-8

# ## Part 2
# #### Load training file and create dictionary containing the occurrences of each annotation in the training set

# In[242]:

def load_original_train():
    with open("train", encoding="utf-8") as file:
        train_list = file.readlines()
        train_list = [x.strip() for x in train_list]
        return train_list
    
train_list = load_original_train()


# In[243]:

annotationDict = {"O": 0, "I-positive": 0, "B-positive": 0, "I-negative":0, "B-negative": 0, "I-neutral":0, "B-neutral":0}

for i in range(len(train_list)):
    annotated_word = train_list[i]
    each_word = annotated_word.split(" ")
    if (len(each_word) == 2):
        if each_word[1] == "O":
            annotationDict["O"] += 1
        elif each_word[1] == "I-positive":
            annotationDict["I-positive"] += 1
        elif each_word[1] == "B-positive":
            annotationDict["B-positive"] += 1
        elif each_word[1] == "I-negative":
            annotationDict["I-negative"] += 1
        elif each_word[1] == "B-negative":
            annotationDict["B-negative"] += 1
        elif each_word[1] == "I-neutral":
            annotationDict["I-neutral"] += 1
        elif each_word[1] == "B-neutral":
            annotationDict["B-neutral"] += 1

# print(annotationDict)


# #### (a) Write a function that estimates the emission parameters from the training set using MLE

# In[244]:

def estimate_emission(word, tag):
    countTop = 0
    countBottom = annotationDict[tag]
    for i in range(len(train_list)):
        annotated_word = train_list[i]
        each_word = annotated_word.split(" ")
        if len(each_word) == 2:
            if (each_word[0] == word and each_word[1] == tag):
                countTop += 1

#         print(countTop)
#         print(countBottom)
    return countTop/countBottom

estimate_emission("service", "O")


# #### (b) Replace the words that appear less than k times in the training set with a special token #UNK# before training. This leads to a “modified training set”.

# In[245]:

import datetime

def modify_trainingset(k):
        wordCountDict = {}
        for i in range(len(train_list)):
            annotated_word = train_list[i]
            each_word = annotated_word.split(" ")
            if each_word[0] in wordCountDict:
                wordCountDict[each_word[0]] += 1
            else:
                wordCountDict[each_word[0]] = 1
        
       
        wordToBeReplacedDict = {}
        wordToBeReplacedList = []
        for key in wordCountDict:
            if wordCountDict[key] < k:
                wordToBeReplacedList.append(key)

        start = datetime.datetime.now()
        for i in range(len(train_list)): 
            annotated_word = train_list[i]
            each_word = annotated_word.split(" ")
            if each_word[0] in wordToBeReplacedList:
                train_list[i] = "#UNK# " + each_word[1]


        modifiedTotalTrainSet = "\n".join(train_list)

        f = open("modified_train","w+")
        f.write(modifiedTotalTrainSet)
                            
        end = datetime.datetime.now()
        elapsed = end - start
        # print(elapsed)

modify_trainingset(3)


# #### Loads the modified training set into a list, and count the occurences of each word in the list and store it into a dictionary called modifiedWordDict

# In[246]:

def load_modified_train_file():
    with open("modified_train", encoding="utf-8") as file:
        modified_train_list = file.readlines()
        modified_train_list = [x.strip() for x in modified_train_list]
        return modified_train_list
        
modified_train_list = load_modified_train_file()


# In[247]:

def get_word_dict():
    modifiedWordDict = {}        
    for i in range(len(modified_train_list)):
        annotated_word = modified_train_list[i]
        each_word = annotated_word.split(" ")
        if each_word[0] in modifiedWordDict:
            modifiedWordDict[each_word[0]] += 1
        else:
            modifiedWordDict[each_word[0]] = 1

    return modifiedWordDict
    
modifiedWordDict = get_word_dict()


# In[248]:

def count_word_tag_pair_dict():
    word_tag_pair_dict= {}
    for word_tag_pair in modified_train_list:
        if word_tag_pair_dict.get(word_tag_pair) == None:
            word_tag_pair_dict[word_tag_pair] = 1
        else:
            word_tag_pair_dict[word_tag_pair] += 1
    return word_tag_pair_dict

    
wordTagPairDict = count_word_tag_pair_dict()


# #### Implement a fixed emission function

# In[249]:

def estimate_emission_fix(word, tag):
        countTop = 0
        countBottom = annotationDict[tag]
        if word in modifiedWordDict:
            # do nothing
            word = word
            
        else:
            word = "#UNK#"

        for i in range(len(modified_train_list)):
            annotated_word = modified_train_list[i]
            each_word = annotated_word.split(" ")
            if (each_word[0] == word and each_word[1] == tag):
                countTop += 1
        return countTop/countBottom

estimate_emission_fix("I", "O")


# #### Loop through the words once and store the emission value of each word tag pair into a giant dictionary called giantEmissionDict

# In[267]:

def store_estimate_emission_fix():
    giantEmissionDict = {}
    for word_tag_pair in modified_train_list:
        if word_tag_pair != "":
            word_tag_pair_list = word_tag_pair.split(" ")
            countTop = wordTagPairDict[word_tag_pair]
            countBottom = annotationDict[word_tag_pair_list[len(word_tag_pair_list)-1]]
            emission = countTop/countBottom
            if giantEmissionDict.get(word_tag_pair) == None:
                giantEmissionDict[word_tag_pair] = countTop/countBottom
    return giantEmissionDict
    
giantEmissionDict = store_estimate_emission_fix()
# print(giantEmissionDict)


# #### (c) Implement a simple sentiment analysis system that produces the tag

# In[251]:

def sentiment_analysis(word):
    all_tags = ["O", "B-positive", "I-positive", "B-negative", "I-negative", "B-neutral", "I-neutral"]
    scoreDict = {}
    # if word never appeared before, let it be #UNK#
    if word in modifiedWordDict:
        word = word
    else:
        word = "#UNK#"
    for tag in all_tags:
        possible_word_tag = word + " " + tag
        if giantEmissionDict.get(possible_word_tag) == None:
            scoreDict[tag] = 0
        else:
            scoreDict[tag] = giantEmissionDict[possible_word_tag]
            
    maximumScore = max(scoreDict.values())
    return max(scoreDict, key=scoreDict.get)

   
sentiment_analysis("ambience")


# #### Loads dev.in file and store as dev_in_list

# In[252]:

def load_files():
    with open("dev.in", encoding="utf-8") as file:
        dev_in_list = file.readlines()
        dev_in_list = [x.strip() for x in dev_in_list]
        return dev_in_list
        
dev_in_list = load_files()


# In[253]:

import datetime
def predict_sentiment():
    wordDict = {}
    f = open("dev.p2.out","w+")
    giantStringToBeWritten = ""
    start = datetime.datetime.now()
    for i in range(len(dev_in_list)):
        each_word = dev_in_list[i]
        if each_word in wordDict:
            tag = wordDict[each_word]
            giantStringToBeWritten += each_word +  " " + tag + "\n"

        else:
            if each_word != "":
                tag = sentiment_analysis(each_word)
                wordDict[each_word] = tag 
                giantStringToBeWritten += each_word +  " " + tag + "\n"

            else:
                giantStringToBeWritten += " \n"


    f.write(giantStringToBeWritten)
    end = datetime.datetime.now()
    elapsed = end - start
    # print(elapsed)

print("started sentiment prediction using emission")
predict_sentiment()
print("finished sentiment prediction using emission")

# ### Part 3 
# #### Function to estimate Transition Parameters using MLE

# #### Creating a giant_each_tweet_tag that contains only the states of each sentence

# In[254]:

def separate_tweets():
    countStart = 1
    # counting starts
    giant_each_tweet = []
    each_tweet = []
    totalStateNumber = {}
    for i in range(len(modified_train_list)):
        annotated_word = modified_train_list[i]
        each_word = annotated_word.split(" ")
        if each_word == ['']:
            countStart += 1
            giant_each_tweet.append(each_tweet)
            each_tweet = []

        else:
            each_tweet.append(each_word[len(each_word)-1])
    for i in range(len(giant_each_tweet)):
        giant_each_tweet[i].insert(0, 'START')
        giant_each_tweet[i].append('STOP')
    
    for each_tweet in giant_each_tweet:
        for tag in each_tweet:
            if totalStateNumber.get(tag) == None:
                totalStateNumber[tag] = 1
            else:
                totalStateNumber[tag] += 1

    return giant_each_tweet, countStart, totalStateNumber
giantTaggedEachTweet, countStart, totalStateNumber = separate_tweets()
# print(totalStateNumber)


# In[266]:

def store_estimate_transition():
    giantTransitionDict = {}
    finalGiantTransitionDict = {}
    for taggedEachTweet in giantTaggedEachTweet:
        for i in range(len(taggedEachTweet) - 1):
            if giantTransitionDict.get(taggedEachTweet[i]+" "+taggedEachTweet[i+1]) == None:
                giantTransitionDict[taggedEachTweet[i]+" "+taggedEachTweet[i+1]] = 1
            else:
                giantTransitionDict[taggedEachTweet[i]+" "+taggedEachTweet[i+1]] += 1
    for eachTransition in giantTransitionDict:
        eachTransitionList = eachTransition.split(" ")
        startState = eachTransitionList[0]
        countTop = giantTransitionDict[eachTransition]
        countBottom = totalStateNumber[startState]
        transitionParam = countTop/countBottom
        finalGiantTransitionDict[eachTransition] = transitionParam
        
    return finalGiantTransitionDict
    
giantTransitionDict = store_estimate_transition()
# print(giantTransitionDict)


# In[261]:

def split_into_sentences(dev_in_list):
    sentences = []
    each_sentence = []
    for word in dev_in_list:
        if word == "":
            sentences.append(each_sentence)
            each_sentence = []
        else:
            each_sentence.append(word)
        
    return sentences
sentences = split_into_sentences(dev_in_list)


# ## Viterbi

# In[262]:

def find_path_backward(piList, sentence):
    path = []
    all_tags=["O","B-negative","B-positive","B-neutral","I-negative","I-positive","I-neutral"]
    if len(piList) == len(sentence):
#         print('sentence length and pilist length are same, move on')
        for i in range(len(piList)):
            # select the max of the last layer, use the max to find the previous value
            if i == 0: #its the start, i = 0 comes here
                lastLayer = piList[len(piList) - (i+1)]
                maxPiTag = max(lastLayer, key=lastLayer.get) 
                path.append('end')
            elif i == 1:
                path.append(maxPiTag)
            else: #i>0 comes here
                currentLayer = piList[len(piList) - (i+1)] #2nd last layer onwards
                targetedTag = path[i-1]
                scoreDict = {}
                for tag in currentLayer: 
                    if giantTransitionDict.get(tag + " " + targetedTag) != None:
                        scoreOfTag = giantTransitionDict[tag + " " + targetedTag] * currentLayer[tag]
                        scoreDict[tag] = scoreOfTag
                    
                bestScoreTag = max(scoreDict, key=scoreDict.get)
                path.append(bestScoreTag)
    singleStringToBeWritten = ""
    for j in range(len(sentence)):
        if sentence[j] == "":
            singleStringToBeWritten += "\n"
        else:
            singleStringToBeWritten += sentence[j] + " " + path[len(path)-(j+1)] + "\n"
    return singleStringToBeWritten

# expected: O, B-pos, I-pos, I-pos, I-pos, O, O, O
piList4 = [{'O': 0.10952242946302841, 'B-negative': 0.0019577571933592874, 'B-positive': 0.010624504726325918, 'B-neutral': 0.0009040105193951349}, {'O': 3.108819999877843e-05}, {'B-negative': 1.212411803650987e-09, 'B-positive': 3.5875587995634343e-09}, {'B-positive': 4.751733509355542e-11}]
sentence2 = ['AVOID', 'THE', 'PLACE', '']
# find_path_backward(piList4, sentence2)


# In[263]:

def viterbi(sentence):
    all_tags=["O","B-negative","B-positive","B-neutral","I-negative","I-positive","I-neutral"]
    sentence.append("") #so we can detect the end
    beginning = True
    piList = []
#     for word in sentence:
    for i in range(len(sentence)):
        word = sentence[i]
        if word in modifiedWordDict:
            word = word
        else:
            word = "#UNK#"
        
        piLayer = {}
        if beginning: #base case
            for tag in all_tags:
                if giantTransitionDict.get("START" + " " + tag) != None and giantEmissionDict.get(word + " " + tag) != None:
                    pi = giantTransitionDict["START" + " " + tag]*giantEmissionDict[word + " " + tag]
                    piLayer[tag] = pi
            piList.append(piLayer)
            beginning = False
        else: #recursive
            previousLayer = piList[len(piList) - 1] #last entry of piList
            if word == "": #end of sentence
                tempLayer = {} 
                for previous_tag in previousLayer:
                    if giantTransitionDict.get(previous_tag + " " + "STOP") != None:
                        pi = previousLayer[previous_tag]*giantTransitionDict[previous_tag + " " + "STOP"] #new score
                        tempLayer[previous_tag] = pi
                if len(tempLayer) != 0:
                    maxPiTag = max(tempLayer, key=tempLayer.get)
                    maximumPiScore = max(tempLayer.values())
                    lastLayer = {}
                    lastLayer[maxPiTag] = maximumPiScore
                    piList.append(lastLayer)
                    singleStringToBeWritten = find_path_backward(piList, sentence)
                    return singleStringToBeWritten
                else:
                    return "ERROR"

            else: #if not end of sentence
                for tag in all_tags:
                    tempLayer = {}
                    for previous_tag in previousLayer:
                        if giantTransitionDict.get(previous_tag + " " + tag) != None and giantEmissionDict.get(word + " " + tag) != None:
                            pi = previousLayer[previous_tag]*giantTransitionDict[previous_tag + " " + tag]*giantEmissionDict[word + " " + tag]
                            tempLayer[previous_tag] = pi
                    # what if the length is 0? it will never transit to the tag
                    if len(tempLayer) != 0:
                        # pick the max here
                        maxPiTag = max(tempLayer, key=tempLayer.get) 
                        maxPiScore = max(tempLayer.values())
                        piLayer[tag] = maxPiScore
                piList.append(piLayer)

print("started viterbi")
giantStringToBeWritten = ""
for sentence in sentences:
# singleStringToBeWritten = viterbi(['AVOID', 'THE', 'PLACE', ''])
    singleStringToBeWritten = viterbi(sentence)
    giantStringToBeWritten += singleStringToBeWritten
# print(giantStringToBeWritten)
f = open("dev.p3.out","w+")
f.write(giantStringToBeWritten)
print("finished viterbi")

# In[ ]:

def combine_path(best_path, sentence):
    result = ""
    sentence.append("")
    for i in range(len(sentence)):
        if (sentence[i] == ''):
            result += " \n"
        else:
            result += sentence[i] + " " + best_path[i] + "\n"
    return result


# In[ ]:

def max_marginal(sentence):
    opti_path = []
    all_tags=["O","B-negative","B-positive","B-neutral","I-negative","I-positive","I-neutral"]
    forward = {}
    alpha_base = {}
    for u in all_tags:
        if giantTransitionDict.get("START" + " " + u) != None:
            alpha_base[u] = giantTransitionDict["START" + " " + u]
        else:
            alpha_base[u] = 0.0
    forward[0] = alpha_base
    for i in range(len(sentence)): #0 ... 7
        if sentence[i] in modifiedWordDict:
            sentence[i] = sentence[i]
        else:
            sentence[i] = "#UNK#"
            
        tempAlpha = {}
        for u in all_tags:
            alpha = 0.0
            for v in all_tags:
                a = 0.0
                b = 0.0
                if giantTransitionDict.get(v + " " + u) != None:
                    a = giantTransitionDict[v + " " + u]
                if giantEmissionDict.get(sentence[i] + " " + v) != None:
                    b = giantEmissionDict[sentence[i] + " " + v]
                alpha += float(forward[i][v] * a * b)
            tempAlpha[u] = alpha
        forward[i + 1] = tempAlpha
    
    lastWord = sentence[len(sentence)-1] #last word 
    if lastWord in modifiedWordDict:
        lastWord = lastWord
    else:
        lastWord = "#UNK#"
    backwards = {} 
    beta_base = {}
    for u in all_tags:
        transition_v = 0.0 
        emission_v = 0.0 
        if giantTransitionDict.get(u + " " + "STOP") != None:
            transition_v = giantTransitionDict[u + " " + "STOP"]
        if giantEmissionDict.get(lastWord + " " + u) != None:
            emission_v = giantEmissionDict[lastWord + " " + u]
        beta_base[u] = float(transition_v * emission_v)
    backwards[len(sentence)] = beta_base #key is 8
    for i in range(len(sentence), 0, -1): #8, .... 1
        i = i -1
        if sentence[i] in modifiedWordDict:
            sentence[i] = sentence[i]
        else:
            sentence[i] = "#UNK#"
            
        tempBeta = {}
        for u in all_tags:
            beta = 0.0
            for v in all_tags:
                a = 0.0
                b = 0.0
                if giantTransitionDict.get(u + " " + v) != None:
                    a = giantTransitionDict[u + " " + v]
                if giantEmissionDict.get(sentence[i] + " " + u) != None:
                    b = giantEmissionDict[sentence[i] + " " + u]
                beta += float(backwards[i+1][v]*a*b)                
            tempBeta[u] = beta
        backwards[i] = tempBeta
    for j in range(len(sentence)):
        temp = {}
        for u in all_tags:
            temp[u] = forward[j][u] * backwards[j][u]
        opti_path.append(max(temp,key=temp.get))
    return opti_path

# max_marginal(["The", "tuna", "and", "wasabe", "potatoes", "are", "excellent", "."])
print("started max marginal")
giantStringToBeWritten = ""
for sentence in sentences:
    best_path = max_marginal(sentence)
    singleStringToBeWritten = combine_path(best_path, sentence)
    giantStringToBeWritten += singleStringToBeWritten
f = open("dev.p4.out","w+")
f.write(giantStringToBeWritten)
print("finished max marginal")


def store_second_order_transition():
    transitionCount = {}
    startStateCount = {}
    giantSecondTransitionDict = {}        
    for taggedEachTweet in giantTaggedEachTweet:
        for i in range(len(taggedEachTweet) - 1):
            if startStateCount.get(taggedEachTweet[i]+" "+taggedEachTweet[i+1]) == None:
                startStateCount[taggedEachTweet[i]+" "+taggedEachTweet[i+1]] = 1
            else:
                startStateCount[taggedEachTweet[i]+" "+taggedEachTweet[i+1]] += 1
                
        for i in range(len(taggedEachTweet) - 2):
            if transitionCount.get(taggedEachTweet[i]+" "+taggedEachTweet[i+1] + " " + taggedEachTweet[i+2]) == None:
                transitionCount[taggedEachTweet[i]+" "+taggedEachTweet[i+1]+ " "+ taggedEachTweet[i+2]] = 1
            else:
                transitionCount[taggedEachTweet[i]+" "+taggedEachTweet[i+1]+ " "+ taggedEachTweet[i+2]] += 1
    for eachTransition in transitionCount:
        eachTransitionList = eachTransition.split(" ") #START O O
        startState = eachTransitionList[0] + " " + eachTransitionList[1] # START O
        countTop = transitionCount[eachTransition]
        countBottom = startStateCount[startState]
        transitionParam = countTop/countBottom
        giantSecondTransitionDict[eachTransition] = transitionParam

    return giantSecondTransitionDict

giantSecondTransitionDict = store_second_order_transition()
# print(giantSecondTransitionDict)

def backtrack_second_order(piList, sentence):
    path = []
    singleStringToBeWritten = ""
    all_tags=["O","B-negative","B-positive","B-neutral","I-negative","I-positive","I-neutral"]
        
    if len(piList) == len(sentence):
        for i in range(len(piList)-1):
            if i == 0:
                lastLayer = piList[len(piList)- (i+1)]
                maxPiTag = max(lastLayer, key=lastLayer.get)
                maxPiTagList = maxPiTag.split(" ")
                path.append(maxPiTagList[len(maxPiTagList)-1])
                path.append(maxPiTagList[0])

            else:
                currentLayer = piList[len(piList) - (i+1)] #2nd last layer onwards
                targetedTag = path[i-1]
                scoreDict = {}
                for tags in currentLayer:
                    if giantSecondTransitionDict.get(tags + " " + targetedTag) != None:
                        scoreOfTag = currentLayer[tags]*giantSecondTransitionDict[tags + " " + targetedTag]
                        scoreDict[tags] = scoreOfTag
                bestScoreTag = max(scoreDict, key=scoreDict.get)
                bestScoreTagList = bestScoreTag.split(" ")
                path.append(bestScoreTagList[0])
                if bestScoreTagList[0] == "START":
                    path.append(bestScoreTagList[len(bestScoreTagList)-1])
    path.remove("STOP")
    path.remove("START")
    for j in range(len(sentence)):
        if sentence[j] == "":
            singleStringToBeWritten += "\n"
        else:
            singleStringToBeWritten += sentence[j] + " " + path[len(path)-(j+1)] + "\n"
    return singleStringToBeWritten
# piList = [{'O': 0.10952242946302841, 'B-negative': 0.0019577571933592874, 'B-positive': 0.010624504726325918, 'B-neutral': 0.0009040105193951349}, {'START O': 0.00024221306515862052, 'START B-negative': 0.0001000958565025801, 'START B-positive': 0.0012160526263888527, 'START B-neutral': 1.9652402595546413e-05}, {'O B-negative': 2.5578783356529717e-08, 'O B-positive': 9.94562791068425e-08}, {'B-positive STOP': 1.413233095656732e-09}]
# sentence = ['AVOID', 'THAT', 'PLACE', '']
# piList2 = [{'B-positive': 0.00014504443312390332, 'B-neutral': 8.21827744904668e-05}, {'START B-positive': 1.701560172955939e-06}, {'B-positive O': 7.4384900942866305e-09}, {'O O': 2.2814373537614629e-10}, {'O O': 1.3952466906739993e-13}, {'O O': 2.5096579686371738e-17}, {'O O': 1.375919532845953e-18}, {'O STOP': 1.0482944960703226e-19}]
# sentence2 = ['lobster', 'was', 'good', ',', 'nothing', 'spectacular', '.', '']
# backtrack_second_order(piList, sentence)
# backtrack_second_order(piList2, sentence2)
def second_order_viterbi(sentence):
    all_tags=["O","B-negative","B-positive","B-neutral","I-negative","I-positive","I-neutral"]
    sentence.append("") #so we can detect the end
    piList = []
    if len(sentence) == 2:
        resultToBeReturned = sentence[0] + " " + "O"
        return resultToBeReturned
    for i in range(len(sentence)):
        word = sentence[i]
        if word in modifiedWordDict:
            word = word
        else:
            word = "#UNK#"
            
        piLayer = {}
        
        if i == 0:
            for tag in all_tags:
                if giantTransitionDict.get("START" + " " + tag) != None and giantEmissionDict.get(word + " " + tag) != None:
                    pi = giantTransitionDict["START" + " " + tag]*giantEmissionDict[word + " " + tag]
                    piLayer[tag] = pi
            piList.append(piLayer)
        
        elif i == 1:
            previousLayer = piList[len(piList)-1]
            for tag in all_tags:
                tempLayer = {}
                for previous_tag in previousLayer:
                    if giantSecondTransitionDict.get("START" + " " + previous_tag + " " + tag) != None and giantEmissionDict.get(word + " " + tag) != None:
                        pi = previousLayer[previous_tag]*giantSecondTransitionDict["START" + " " + previous_tag + " " + tag]*giantEmissionDict[word + " " + tag]
                        tempLayer["START" + " " + previous_tag] = pi
                if len(tempLayer) != 0:
                    maxPiTag = max(tempLayer, key=tempLayer.get) 
                    maxPiScore = max(tempLayer.values())
                    piLayer[maxPiTag] = maxPiScore
            piList.append(piLayer)
            
        else:
            
            previousLayer = piList[len(piList)-1]
#             print(previousLayer)
            if word == "":
#                 print(piList)
                tempLayer = {}
                for previous_tags in previousLayer:
                    previous_tags_list = previous_tags.split(" ")
                    if giantSecondTransitionDict.get(previous_tags + " " + "STOP") != None:
                        pi = previousLayer[previous_tags]*giantSecondTransitionDict[previous_tags + " " + "STOP"]
                        tagOneLayerBefore = previous_tags_list[len(previous_tags_list)-1]
                        tempLayer[tagOneLayerBefore+ " " + "STOP"] = pi
                if len(tempLayer) != 0:
                    maxPiTag = max(tempLayer, key=tempLayer.get) 
                    maxPiScore = max(tempLayer.values())
                    piLayer[maxPiTag] = maxPiScore
#                 else:
#                     piLayer[]
                piList.append(piLayer)
                singleStringToBeWritten = backtrack_second_order(piList, sentence)
#                 print(singleStringToBeWritten)
                return singleStringToBeWritten
            else:
                for tag in all_tags:
                    tempLayer = {}
                    for previous_tags in previousLayer:
                        previous_tags_list = previous_tags.split(" ")
                        if giantSecondTransitionDict.get(previous_tags + " " + tag) != None and giantEmissionDict.get(word + " " + tag) != None:
                            pi = previousLayer[previous_tags]*giantSecondTransitionDict[previous_tags + " " + tag]*giantEmissionDict[word + " " + tag]
                            tagOneLayerBefore = previous_tags_list[len(previous_tags_list)-1]
                            tempLayer[tagOneLayerBefore+ " " + tag] = pi
#                         else:
#                             print("problem from %s  to %s" % (previous_tags, tag))
#                             print(word)
#                             print(giantSecondTransitionDict.get(previous_tags + " " + tag))
#                             # riz only has emission from B - positive
#                             # no transition from I-neutral O to B-positive
#                             print(giantSecondTransitionDict)
#                             print(giantEmissionDict.get(word + " " + tag))
                    if len(tempLayer) != 0:
                        maxPiTag = max(tempLayer, key=tempLayer.get) 
                        maxPiScore = max(tempLayer.values())
                        piLayer[maxPiTag] = maxPiScore
                    
                # any other way to fix this??
                if len(piLayer) == 0:
                    for previous_tags in previousLayer:
                        previous_tags_list = previous_tags.split(" ")

                        piLayer[previous_tags_list[len(previous_tags_list)-1] + " " + "O"] = 0

                piList.append(piLayer)

sentences = split_into_sentences(dev_in_list)
print("started viterbi with second order")
giantStringToBeWritten = ""    
for sentence in sentences:
    singleStringToBeWritten = second_order_viterbi(sentence)
    giantStringToBeWritten += singleStringToBeWritten
# print(giantStringToBeWritten)
f = open("dev.p5.out","w+")
f.write(giantStringToBeWritten)
print("finished viterbi with second order")

# second_order_viterbi(['des', 'plats', 'sans', 'passion', ',', 'vaisselle', 'cassée', ',', 'plat', 'du', 'jour', '(', 'riz', 'crustacés', 'en', 'gros', ')', 'très', 'sec', ',', 'sans', 'sauce', ',', 'gambas', 'pas', 'fraîches', ',', 'une', 'pauvre', 'décoration', 'superflue', '...', 'bref', 'aucun', 'goût', '.'])
# second_order_viterbi(["lobster", "was", "good", ",", "nothing","spectacular","."])