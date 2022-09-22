#!/usr/bin/env python
# coding: utf-8

# # FINANCIAL FRAUD DETCTION USING PYSPARK

# In[50]:


#load_ext nb_black
from pyspark.sql import SparkSession
#session all related to df not rdd
import pyspark.sql.functions as F
import pyspark.sql.types as T

spark = SparkSession.builder.getOrCreate()


# In[51]:


spark


# In[52]:


df = spark.read.csv("financial.csv", inferSchema=True, header=True)


# In[53]:


df.printSchema()


# In[54]:


df.show(2)


# In[55]:


df = df.select("type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud")


# In[56]:


df.show(2)


# In[57]:


df.printSchema()


# In[58]:


df.count() , len(df.columns)


# In[59]:


df.select('amount','oldbalanceOrg','newbalanceOrig','isFraud').describe().show()


# In[60]:


# null values in each column
data_agg = df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns])
data_agg.show()


# In[12]:


# value counts of Type column
df.groupBy('type').count().show()


# In[61]:


train, test = df.randomSplit([0.7, 0.3], seed=7)


# In[62]:


print(f"Train set length: {train.count()} records")
print(f"Test set length: {test.count()} records")


# In[63]:


train.show(2)


# In[16]:


train.dtypes


# In[64]:


train.show(2)


# In[65]:


catCols = [x for (x, dataType) in train.dtypes if dataType == "string"]
numCols = [ x for (x, dataType) in train.dtypes if (dataType == "double") ]
#numCols = [ x for (x, dataType) in train.dtypes if ((dataType == "double") & (x != "isFraud")) ]
#skip the "isFraud" but 


# In[66]:


print(numCols)
print(catCols)


# In[67]:


train.agg(F.countDistinct("type")).show()


# In[68]:


train.groupBy("type").count().show()


# In[69]:


from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
)


# In[70]:


df = df.select("type","isFraud")


# In[71]:


#catCols are the cols with string
string_indexer = [
    StringIndexer(inputCol=x, outputCol=x + "_StringIndexer", handleInvalid="skip")
    for x in catCols
]


# In[25]:


string_indexe=string_indexer[0].fit(df).transform(df)
string_indexe.show()


# In[72]:


one_hot_encoder = [
    OneHotEncoder(
        inputCols=[f"{x}_StringIndexer" for x in catCols],
        outputCols=[f"{x}_OneHotEncoder" for x in catCols],
    )
]


# In[73]:


from pyspark.ml.feature import VectorAssembler


# In[74]:


assemblerInput = [x for x in numCols]
assemblerInput += [f"{x}_OneHotEncoder" for x in catCols]
assemblerInput


# In[75]:


vector_assembler = VectorAssembler(
    inputCols=assemblerInput, outputCol="VectorAssembler_features"
)


# In[76]:


stages = []
stages += string_indexer
stages += one_hot_encoder
stages += [vector_assembler]


# In[77]:


stages


# In[78]:


#%%time
from pyspark.ml import Pipeline

pipeline = Pipeline().setStages(stages)
model = pipeline.fit(train)

pp_df = model.transform(train)


# In[80]:


pp_df.select(
    "type", "amount", "oldbalanceOrg", "newbalanceOrig", "VectorAssembler_features",
).show(truncate=False)


# In[81]:


pp_df.show()


# In[82]:


test.count()


# In[36]:


df_test=test.where(test.isFraud == 1)


# In[37]:


df_test.show()


# In[38]:


from pyspark.ml.classification import LogisticRegression


# In[39]:


data = pp_df.select(
    F.col("VectorAssembler_features").alias("features"),
    F.col("isFraud").alias("label"),
)


# In[40]:


data.show(5, truncate=False)


# In[41]:


get_ipython().run_cell_magic('time', '', 'model = LogisticRegression().fit(data)\ndata=model.transform(data)')


# In[42]:


data.show()


# In[43]:


model = pipeline.fit(df_test)

pp_df_test = model.transform(df_test)


# In[44]:


data_test = pp_df_test.select(
    F.col("VectorAssembler_features").alias("features"),
    F.col("isFraud").alias("label"),
)


# In[45]:


data_test.show(5, truncate=False)


# In[46]:


model = LogisticRegression().fit(data_test)
data=model.transform(data_test)
data.show()


# In[47]:


df.limit


# In[48]:


model.summary.areaUnderROC


# In[49]:


model.summary.pr.show()


# In[ ]:




