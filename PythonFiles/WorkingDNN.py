
# coding: utf-8

# In[2]:


import glob
import skimage
import os
from skimage import io
import matplotlib
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.metrics import f1_score
from skimage.transform import rescale, resize
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
from sklearn.utils import shuffle


# In[ ]:


# random.seed(125)
# random.shuffle(X_cats_dogs)
# random.shuffle(y_cats_dogs)


# In[47]:


cd Desktop


# In[48]:


cd Train_data/


# In[49]:


dogs_races = glob.glob('/media/ficiu/ML/Train_data/Caini/Cu rasa/*')


# In[50]:


len(dogs_races)


# In[51]:


cats_races=glob.glob('/media/ficiu/ML/Train_data/Pisici/Cu rasa/*')


# In[52]:


dogs_without_race=glob.glob('/media/ficiu/ML/Train_data/Caini/Fara rasa/*')


# In[53]:


cats_without_race=glob.glob('/media/ficiu/ML/Train_data/Pisici/Fara rasa/*')


# In[54]:


X_cats_dogs=[]
y_cats_dogs=[]

X_races=[]
y_races=[]
z_races=[]


# In[55]:


for each in dogs_races:
    for i in glob.glob(each+'/*'):
        X_races.append(i)
        y_races.append(each.split('/')[-1])
        z_races.append(1)


# In[56]:


for each in cats_races:
    for i in glob.glob(each+'/*'):
        X_races.append(i)
        y_races.append(each.split('/')[-1])
        z_races.append(0)


# In[57]:


cats_races


# In[58]:


count_dogs=0
count_cats=0
for each in dogs_races:
    for i in glob.glob(each+'/*'):
        X_cats_dogs.append(i)
        y_cats_dogs.append('1')
        count_dogs+=1
        
for each in cats_races:
    for i in glob.glob(each+'/*'):
        X_cats_dogs.append(i)
        y_cats_dogs.append('0')
        count_cats+=1
        
for each in dogs_without_race:
    X_cats_dogs.append(each)
    y_cats_dogs.append('1')
    count_dogs+=1

for each in cats_without_race:
    X_cats_dogs.append(each)
    y_cats_dogs.append('0')
    count_cats+=1

for each in cats_without_race:
    X_cats_dogs.append(each)
    y_cats_dogs.append('0')
    count_cats+=1
for each in cats_races:
    for i in glob.glob(each+'/*'):
        X_cats_dogs.append(i)
        y_cats_dogs.append('0')
        count_cats+=1
    
print(count_dogs-count_cats,count_cats)


# In[59]:


len(X_races),len(y_races),len(z_races)


# In[60]:


len(X_cats_dogs), len(y_cats_dogs)


# In[61]:


X_cats_dogs, y_cats_dogs = shuffle(X_cats_dogs, y_cats_dogs)


# In[62]:


a=0
b=0
for i in y_cats_dogs:
    if(i=='1'):
        a+=1
    else:
        b+=1
print(a,b)


# In[3]:


pic_width=100
pic_height=100
n_inputs = pic_width*pic_height*3
n_hidden1= 4000
n_hidden2= 4000
n_hidden3= 1000
n_hidden4= 300
n_hidden5= 100
n_hidden6= 30
n_outputs = 2


# In[64]:


a,b


# In[4]:


def prepare_batch(batch_size ,startPointInDataset ,X_dataset ,Y_dataset ,input_size):
    x_batch=np.zeros(batch_size*input_size).reshape(batch_size, input_size) #pregatit forma matricei de input
    y_batch=np.zeros(batch_size)   #pregatit forma matricei label
    
    #Daca Batchsizeul + starting position > lungimea datasetului
    #Micesc Batchsizeul si antrenez pe ce o ramas
    if(startPointInDataset+batch_size>len(X_dataset)):
        newBatchSize=len(X_dataset)-startPointInDataset
        x_batch,y_batch=prepare_batch(newBatchSize,startPointInDataset,X_dataset,Y_dataset,input_size)
    else:
        i=0
        while i < batch_size:
            #Citesc poza
            try:
                poza_resized = resize(io.imread(X_cats_dogs[startPointInDataset+i]),(pic_height,pic_width),anti_aliasing=True)
                if(poza_resized.flatten().shape[0]==pic_height*pic_width*3):
                    x_batch[i]=poza_resized.flatten()
                    y_batch[i]=y_cats_dogs[startPointInDataset+i]
                
            except:
                print("POZA: ", X_cats_dogs[startPointInDataset:startPointInDataset+batch_size][i], "Nu e buna")
                if(i>0):
                    x_batch[i]=x_batch[i-1]
                    y_batch[i]=y_batch[i-1]
            i+=1
    return x_batch , y_batch


# Model Construction Phase

# In[5]:


#Input and output placeholders
tf.reset_default_graph
X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")
y=tf.placeholder(tf.int64,shape=(None),name="y")


# In[6]:


#Model hiddenLayers and number of neurons /layer and the type of connection
with tf.name_scope("really_deep_nn"):
    hidden1=fully_connected(X,n_hidden1,scope='hidden1',activation_fn=tf.nn.leaky_relu)
    hidden2=fully_connected(hidden1,n_hidden2,scope='hidden2',activation_fn=tf.nn.leaky_relu)
    hidden3=fully_connected(hidden2,n_hidden3,scope='hidden3',activation_fn=tf.nn.leaky_relu)
    hidden4=fully_connected(hidden3,n_hidden4,scope='hidden4',activation_fn=tf.nn.leaky_relu)
    hidden5=fully_connected(hidden4,n_hidden5,scope='hidden5',activation_fn=tf.nn.leaky_relu)
    hidden6=fully_connected(hidden5,n_hidden6,scope='hidden6',activation_fn=tf.nn.leaky_relu)
    output=fully_connected(hidden6,n_outputs,scope="outputs",activation_fn=None)


# In[7]:


#Loss Function or How the network checks if it got the right answer
with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output)
    loss=tf.reduce_mean(xentropy,name="loss")


# In[8]:


#The algorithm used to adjust all the variables in order to minimise the loss function from above (train)
learning_rate=0.00002
with tf.name_scope("train"):
    optimizer=tf.train.AdamOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)


# In[9]:


#How the network will be evaluated
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(output,y,1)
    accuracy =tf.reduce_mean(tf.cast(correct,tf.float32))


# In[10]:


init =tf.global_variables_initializer()
saver=tf.train.Saver()


# In[ ]:


n_epochs=100
batch_size=100
X_train, X_test, y_train, y_test = train_test_split(X_cats_dogs, y_cats_dogs, test_size=0.2, random_state=10)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.2, random_state=10)
print(len(X_test))
now = time.time()

with tf.Session() as sess:
    saver.restore(sess,"./dnn1.ckpt")
    for epoch in range(n_epochs):
        now=time.time()
        train_accuracy=0
        for iteration in range(len(X_train)//batch_size):
            startPositionForNextBatch= batch_size*iteration
            X_batch, y_batch=prepare_batch(batch_size ,startPositionForNextBatch ,X_train,y_train,n_inputs)
            if(iteration%100!=0):   
                train_accuracy+= sess.run([training_op, accuracy],feed_dict={X:X_batch,y:y_batch})[1]
            else:
                train_accuracy+=sess.run([training_op,accuracy],feed_dict={X:X_batch,y:y_batch})[1]
                print(train_accuracy/iteration)
                print(output.eval(feed_dict={X:X_batch})[0],y_batch[0])

        #Testing accuracy on validation
        test=0
        for i in range(len(X_test)//batch_size):
            poze, labele=prepare_batch(batch_size ,i*batch_size ,X_test,y_test,n_inputs)
            test+=accuracy.eval(feed_dict={X:poze,y:labele})
            print(test/i)
        
                
        print("Epoch: ", epoch, "Train Accuracy= ", train_accuracy/(len(X_train)/batch_size) , "Validation Accuracy= ", test/(len(X_test)/batch_size), " with a time of:", time.time()-now)

        save_path=saver.save(sess,"./dnn1.ckpt")


# In[12]:


poza_daria=io.imread('/home/ficiu/Desktop/cat.jpg')


# In[15]:


ls


# In[ ]:


poza_resized=np.zeros(1*100*100*3).reshape(1, 100*100*3)
poza_resized[0] = resize(poza_daria,(pic_height,pic_width),anti_aliasing=True).flatten()
with tf.Session() as sess:
     saver.restore(sess,"./my_model_final.ckpt")
     print(correct.eval(feed_dict={X:poza_resized}))


# In[ ]:


# Epoch:  0 Train Accuracy=  0.5013940307351563 Validation Accuracy=  0.4735234168845621  with a time of: 2428.9339051246643
# Epoch:  1 Train Accuracy=  0.49930616283124146 Validation Accuracy=  0.47250508791681706  with a time of: 2455.0604062080383
# Epoch:  2 Train Accuracy=  0.5013176457085378 Validation Accuracy=  0.512983702320431  with a time of: 2478.5970554351807
# Epoch:  3 Train Accuracy=  0.5034691686549754 Validation Accuracy=  0.4753054940955459  with a time of: 2455.6275560855865
# Epoch:  4 Train Accuracy=  0.5026798523500281 Validation Accuracy=  0.4979633401828968  with a time of: 2446.111537218094
# Epoch:  5 Train Accuracy=  0.5072247870630198 Validation Accuracy=  0.49490834720994203  with a time of: 2549.2453978061676
# Epoch:  6 Train Accuracy=  0.5084342226268642 Validation Accuracy=  0.48676170939826185  with a time of: 2545.138669729233
# Epoch:  7 Train Accuracy=  0.5093890402970817 Validation Accuracy=  0.5119653756288307  with a time of: 2413.832659959793
# Epoch:  8 Train Accuracy=  0.510687592920458 Validation Accuracy=  0.5310590597070647  with a time of: 2517.605666399002
# Epoch:  9 Train Accuracy=  0.5140358219398778 Validation Accuracy=  0.5068737232029559  with a time of: 2471.785297393799
# Epoch:  10 Train Accuracy=  0.5171039699863119 Validation Accuracy=  0.5267311570418585  with a time of: 2243.544430732727
# Epoch:  11 Train Accuracy=  0.5212287838468819 Validation Accuracy=  0.5318228056741586  with a time of: 2260.661299943924
# Epoch:  12 Train Accuracy=  0.5242460089975296 Validation Accuracy=  0.5257128265566835  with a time of: 2247.320822238922
# Epoch:  13 Train Accuracy=  0.5299112621010109 Validation Accuracy=  0.539714867313622  with a time of: 2242.708364009857
# Epoch:  14 Train Accuracy=  0.5323683276959739 Validation Accuracy=  0.5397148650374771  with a time of: 2250.7277886867523
# Epoch:  15 Train Accuracy=  0.5373461122210745 Validation Accuracy=  0.548370666574315  with a time of: 2257.4932301044464
# Epoch:  16 Train Accuracy=  0.5440807627163121 Validation Accuracy=  0.5549898132105223  with a time of: 2251.711149454117
# Epoch:  17 Train Accuracy=  0.5501152121070222 Validation Accuracy=  0.5654277003042081  with a time of: 2299.68478679657
# Epoch:  18 Train Accuracy=  0.5561751243494446 Validation Accuracy=  0.5623727050551086  with a time of: 2265.840857028961
# Epoch:  19 Train Accuracy=  0.5627060785568454 Validation Accuracy=  0.572046840870696  with a time of: 2253.4909648895264
# Epoch:  20 Train Accuracy=  0.5705355854635046 Validation Accuracy=  0.5723014238713472  with a time of: 2238.7408182621
# Epoch:  21 Train Accuracy=  0.576468190352638 Validation Accuracy=  0.5761201620526809  with a time of: 2260.239249229431
# Epoch:  22 Train Accuracy=  0.584577778270224 Validation Accuracy=  0.5784114022301073  with a time of: 2252.315010547638
# Epoch:  23 Train Accuracy=  0.5933748349558958 Validation Accuracy=  0.5913951094670111  with a time of: 2251.169410467148
# Epoch:  24 Train Accuracy=  0.6022737397096039 Validation Accuracy=  0.5980142561032185  with a time of: 2269.177621126175
# Epoch:  25 Train Accuracy=  0.6091866219336182 Validation Accuracy=  0.6025967432865057  with a time of: 2817.8558905124664
# Epoch:  26 Train Accuracy=  0.6170543230799047 Validation Accuracy=  0.6053971464303749  with a time of: 2673.306975841522
# Epoch:  27 Train Accuracy=  0.6244255182333278 Validation Accuracy=  0.6127800382749612  with a time of: 2377.2644815444946
# Epoch:  28 Train Accuracy=  0.6336045025182615 Validation Accuracy=  0.6193991849111685  with a time of: 2517.263561487198
# Epoch:  29 Train Accuracy=  0.6399444938366168 Validation Accuracy=  0.6237270860589449  with a time of: 2791.3223090171814
# Epoch:  30 Train Accuracy=  0.6462844853446774 Validation Accuracy=  0.6283095724835173  with a time of: 2279.47269654274
# Epoch:  31 Train Accuracy=  0.6458516343563955 Validation Accuracy=  0.6288187415196793  with a time of: 2251.6924073696136
# Epoch:  32 Train Accuracy=  0.6510331133861031 Validation Accuracy=  0.6308553994551696  with a time of: 2252.1026055812836
# Epoch:  33 Train Accuracy=  0.6611923764097368 Validation Accuracy=  0.6390020403017095  with a time of: 2263.786932706833
# Epoch:  34 Train Accuracy=  0.6691873867304834 Validation Accuracy=  0.6466395121120873  with a time of: 2269.442892074585
# Epoch:  35 Train Accuracy=  0.6767877394299311 Validation Accuracy=  0.6537678224734522  with a time of: 2278.369021654129
# Epoch:  36 Train Accuracy=  0.6831404602012076 Validation Accuracy=  0.6588594733818971  with a time of: 2432.652182340622
# Epoch:  37 Train Accuracy=  0.6874816999163584 Validation Accuracy=  0.66140529428383  with a time of: 2308.5474047660828
# Epoch:  38 Train Accuracy=  0.6965079131995284 Validation Accuracy=  0.6682790224032586  with a time of: 2373.931582927704
# Epoch:  39 Train Accuracy=  0.7043628825784893 Validation Accuracy=  0.6710794331342771  with a time of: 2440.4019021987915
# Epoch:  40 Train Accuracy=  0.7099517520212302 Validation Accuracy=  0.6708248433051915  with a time of: 2554.070233106613
# Epoch:  41 Train Accuracy=  0.7066799087113679 Validation Accuracy=  0.6652240309477337  with a time of: 2560.3832080364227
# Epoch:  42 Train Accuracy=  0.7119759637443906 Validation Accuracy=  0.672861508827831  with a time of: 2316.371401786804
# Epoch:  43 Train Accuracy=  0.72121860246479 Validation Accuracy=  0.6764256662846583  with a time of: 2300.19793343544
# Epoch:  44 Train Accuracy=  0.7303212012678123 Validation Accuracy=  0.6723523519311078  with a time of: 2287.2847154140472
# Epoch:  45 Train Accuracy=  0.7367885009259242 Validation Accuracy=  0.6703156818561787  with a time of: 2286.69592833519
# Epoch:  46 Train Accuracy=  0.7456492122747082 Validation Accuracy=  0.6748981697981808  with a time of: 2304.059144973755
# Epoch:  47 Train Accuracy=  0.7515054293675256 Validation Accuracy=  0.67566191348913  with a time of: 2315.986036300659
# Epoch:  48 Train Accuracy=  0.7584183125400663 Validation Accuracy=  0.6787169019097952  with a time of: 2320.397607088089
# Epoch:  49 Train Accuracy=  0.7517345853959079 Validation Accuracy=  0.6878818793112295  with a time of: 2270.347267150879
# Epoch:  50 Train Accuracy=  0.752918560043034 Validation Accuracy=  0.691191448456401  with a time of: 2251.0444536209106
# Epoch:  51 Train Accuracy=  0.7571834146177432 Validation Accuracy=  0.699592662440298  with a time of: 2276.2016429901123
# Epoch:  52 Train Accuracy=  0.7696087778431844 Validation Accuracy=  0.7029022406900487  with a time of: 2290.2707436084747
# Epoch:  53 Train Accuracy=  0.7819832198422458 Validation Accuracy=  0.7146130314062666  with a time of: 2261.736209154129
# Epoch:  54 Train Accuracy=  0.7922188678166159 Validation Accuracy=  0.7179226066211576  with a time of: 2333.298304796219
# Epoch:  0 Train Accuracy=  0.798558859590264 Validation Accuracy=  0.7237780057728411  with a time of: 2255.4047331809998
# Epoch:  1 Train Accuracy=  0.8057518231287335 Validation Accuracy=  0.725305496189599  with a time of: 2227.9408717155457
# Epoch:  2 Train Accuracy=  0.8104367959379314 Validation Accuracy=  0.7326883903103302  with a time of: 2219.7313334941864
# Epoch:  3 Train Accuracy=  0.8027727903586014 Validation Accuracy=  0.7372708797697622  with a time of: 2237.9004395008087
# Epoch:  4 Train Accuracy=  0.8008504226817249 Validation Accuracy=  0.7431262728517264  with a time of: 2226.70755982399
# Epoch:  5 Train Accuracy=  0.7892271061247375 Validation Accuracy=  0.7281059084380473  with a time of: 2225.938397169113
# Epoch:  6 Train Accuracy=  0.787279277113791 Validation Accuracy=  0.7576374773338225  with a time of: 2317.1757423877716
# Epoch:  7 Train Accuracy=  0.8009522705979969 Validation Accuracy=  0.7573828950918862  with a time of: 2659.871309518814
# Epoch:  8 Train Accuracy=  0.8209016018377367 Validation Accuracy=  0.7581466372654054  with a time of: 2413.227341413498
# Epoch:  9 Train Accuracy=  0.8357458386190947 Validation Accuracy=  0.7734215869558804  with a time of: 2427.2332949638367
# Epoch:  10 Train Accuracy=  0.8454595207462327 Validation Accuracy=  0.7759674124101029  with a time of: 2406.5069229602814
# Epoch:  11 Train Accuracy=  0.8528689079766428 Validation Accuracy=  0.7762219916171794  with a time of: 2326.3464155197144
# Epoch:  12 Train Accuracy=  0.8590179365577169 Validation Accuracy=  0.7836048857379105  with a time of: 2301.6406302452087
# Epoch:  13 Train Accuracy=  0.8635246787384288 Validation Accuracy=  0.7808044795591816  with a time of: 2308.437893629074
# Epoch:  14 Train Accuracy=  0.8619715092078929 Validation Accuracy=  0.7746944981655618  with a time of: 2298.9020204544067
# Epoch:  15 Train Accuracy=  0.8492660630365308 Validation Accuracy=  0.7604378789602617  with a time of: 2287.5791568756104
# Epoch:  16 Train Accuracy=  0.8270378975891354 Validation Accuracy=  0.7624745384131818  with a time of: 2316.456241607666
# Epoch:  17 Train Accuracy=  0.8213726462309427 Validation Accuracy=  0.7647657831428978  with a time of: 2301.9176416397095
# Epoch:  18 Train Accuracy=  0.817515180968976 Validation Accuracy=  0.7334521324838494  with a time of: 2343.100418806076
# Epoch:  19 Train Accuracy=  0.8068466805266227 Validation Accuracy=  0.7609470449615642  with a time of: 2704.746750831604
# Epoch:  20 Train Accuracy=  0.8202395949348957 Validation Accuracy=  0.7370162914581065  with a time of: 2624.3215346336365
# Epoch:  21 Train Accuracy=  0.8522323646084532 Validation Accuracy=  0.7657841136280727  with a time of: 2405.2771739959717
# Epoch:  22 Train Accuracy=  0.8798966251580127 Validation Accuracy=  0.7762219946520391  with a time of: 2583.601537704468
# Epoch:  23 Train Accuracy=  0.890323238297892 Validation Accuracy=  0.8011710816755314  with a time of: 2336.6623799800873
# Epoch:  24 Train Accuracy=  0.897070619156636 Validation Accuracy=  0.7955702632483542  with a time of: 2395.5624630451202
# Epoch:  25 Train Accuracy=  0.9040853499419202 Validation Accuracy=  0.7869144609528015  with a time of: 2380.753798007965
# Epoch:  26 Train Accuracy=  0.909597832802459 Validation Accuracy=  0.7787678216236914  with a time of: 2338.683547973633
# Epoch:  27 Train Accuracy=  0.9125386717503153 Validation Accuracy=  0.7787678201062616  with a time of: 2447.23682427406


    
# Accuracy on the test dataset: 0.7162820551639948


# In[ ]:


# Epoch:  0 Train Accuracy=  0.5921941154153199 Validation Accuracy=  0.6194442683047876  with a time of: 2014.5213162899017
# Epoch:  1 Train Accuracy=  0.6239365144136935 Validation Accuracy=  0.6443059720336318  with a time of: 2308.676289319992
# Epoch:  2 Train Accuracy=  0.6373787087195174 Validation Accuracy=  0.655306163288659  with a time of: 2813.3859832286835
# Epoch:  3 Train Accuracy=  0.6481502686801777 Validation Accuracy=  0.6675780539580247  with a time of: 2915.059719800949
# Epoch:  4 Train Accuracy=  0.6598374742872968 Validation Accuracy=  0.6762256020924167  with a time of: 2847.7873351573944
# Epoch:  5 Train Accuracy=  0.6710795730963737 Validation Accuracy=  0.6894512652380601  with a time of: 2843.1716661453247
# Epoch:  6 Train Accuracy=  0.6812407023733912 Validation Accuracy=  0.6971450382432047  with a time of: 2806.670207977295
# Epoch:  7 Train Accuracy=  0.6911983544021227 Validation Accuracy=  0.7052839086584378  with a time of: 2864.784115552902
# Epoch:  8 Train Accuracy=  0.7010797009941017 Validation Accuracy=  0.7103707006308569  with a time of: 2812.476646900177
# Epoch:  9 Train Accuracy=  0.7144583080676404 Validation Accuracy=  0.7231512716723509  with a time of: 1881.2878868579865
# Epoch:  10 Train Accuracy=  0.725992905795396 Validation Accuracy=  0.73268900591002  with a time of: 2029.8346664905548
# Epoch:  11 Train Accuracy=  0.7384812998034578 Validation Accuracy=  0.7334520249522299  with a time of: 1957.8744022846222
# Epoch:  12 Train Accuracy=  0.7516055604492912 Validation Accuracy=  0.7422267458326235  with a time of: 1969.6813855171204
# Epoch:  13 Train Accuracy=  0.7650859051236586 Validation Accuracy=  0.7582501423080734  with a time of: 2079.28089261055
# Epoch:  14 Train Accuracy=  0.7782991852416542 Validation Accuracy=  0.7647358045458543  with a time of: 1901.3931438922882
# Epoch:  15 Train Accuracy=  0.7899228046398025 Validation Accuracy=  0.7724295760350164  with a time of: 2067.5796139240265
# Epoch:  16 Train Accuracy=  0.8025892411548382 Validation Accuracy=  0.7754816533408435  with a time of: 2100.779406309128
# Epoch:  17 Train Accuracy=  0.8160950217833345 Validation Accuracy=  0.7802505221651583  with a time of: 1846.2456285953522
# Epoch:  18 Train Accuracy=  0.8221230250022352 Validation Accuracy=  0.7985629787992035  with a time of: 1845.4435501098633
# Epoch:  19 Train Accuracy=  0.8312794860906888 Validation Accuracy=  0.8057480744641848  with a time of: 1846.8347253799438
# Epoch:  20 Train Accuracy=  0.831699157199906 Validation Accuracy=  0.8118522256648781  with a time of: 1853.3317449092865
# Epoch:  21 Train Accuracy=  0.8337975135798031 Validation Accuracy=  0.8140141138354631  with a time of: 1853.780173778534
# Epoch:  22 Train Accuracy=  0.8481680709406425 Validation Accuracy=  0.8256501525236857  with a time of: 1845.0115020275116
# Epoch:  23 Train Accuracy=  0.8619409155191023 Validation Accuracy=  0.8320086458054214  with a time of: 1856.4795513153076
# Epoch:  24 Train Accuracy=  0.8754085435808245 Validation Accuracy=  0.8356329845504387  with a time of: 1856.8784039020538
# Epoch:  25 Train Accuracy=  0.8847557653049185 Validation Accuracy=  0.838367136055192  with a time of: 1865.3127417564392
# Epoch:  26 Train Accuracy=  0.8888380205917935 Validation Accuracy=  0.8507026115815759  with a time of: 1878.6713659763336
# Epoch:  27 Train Accuracy=  0.8838401179529358 Validation Accuracy=  0.8275577012661977  with a time of: 1879.008945465088
# Epoch:  28 Train Accuracy=  0.8817544804597157 Validation Accuracy=  0.8388122330862847  with a time of: 1862.3691236972809
# Epoch:  29 Train Accuracy=  0.8854170645615725 Validation Accuracy=  0.8493673261732323  with a time of: 1869.5221998691559
# Epoch:  30 Train Accuracy=  0.8926023430524895 Validation Accuracy=  0.862910917772919  with a time of: 1856.0632095336914
# Epoch:  31 Train Accuracy=  0.9007795718307071 Validation Accuracy=  0.8617028005626293  with a time of: 1866.1150195598602
# Epoch:  32 Train Accuracy=  0.9086897377201758 Validation Accuracy=  0.8560437471510199  with a time of: 1871.0322544574738
# Epoch:  33 Train Accuracy=  0.9067058374487791 Validation Accuracy=  0.8599224249451145  with a time of: 1859.0510869026184
# Epoch:  34 Train Accuracy=  0.9096435352891 Validation Accuracy=  0.8684428045024569  with a time of: 1860.4828052520752
# Epoch:  35 Train Accuracy=  0.9170450073377441 Validation Accuracy=  0.8667260122259778  with a time of: 1852.0192258358002
# Epoch:  36 Train Accuracy=  0.9205168319409762 Validation Accuracy=  0.8725122402785669  with a time of: 1859.2039065361023
# Epoch:  37 Train Accuracy=  0.9261760339880899 Validation Accuracy=  0.884339036948805  with a time of: 1875.1206061840057
# Epoch:  38 Train Accuracy=  0.9280200439402817 Validation Accuracy=  0.8763273351106214  with a time of: 1848.1273674964905
# Epoch:  39 Train Accuracy=  0.9334248998628941 Validation Accuracy=  0.8937496041254069  with a time of: 1860.2863914966583
# Epoch:  40 Train Accuracy=  0.9389569272180346 Validation Accuracy=  0.9006167743683103  with a time of: 1858.4054398536682
# Epoch:  41 Train Accuracy=  0.9409662629129527 Validation Accuracy=  0.8933680938463107  with a time of: 1933.1858048439026
# Epoch:  42 Train Accuracy=  0.9413096297982341 Validation Accuracy=  0.8998537560840916  with a time of: 1900.534880399704
# Epoch:  43 Train Accuracy=  0.9475411112508344 Validation Accuracy=  0.9002988485672366  with a time of: 1862.8642683029175
# Epoch:  44 Train Accuracy=  0.9495885981481077 Validation Accuracy=  0.8927322433811502  with a time of: 1864.0862522125244
# Epoch:  45 Train Accuracy=  0.9495250117496343 Validation Accuracy=  0.9123799858025345  with a time of: 1859.9062051773071
# Epoch:  46 Train Accuracy=  0.9509111976468294 Validation Accuracy=  0.9170852693908355  with a time of: 1857.5071768760681
# Epoch:  47 Train Accuracy=  0.9517251062609676 Validation Accuracy=  0.9062758334648543  with a time of: 2020.2043707370758
# Epoch:  48 Train Accuracy=  0.9523228199529598 Validation Accuracy=  0.8954028100288853  with a time of: 2022.608127117

