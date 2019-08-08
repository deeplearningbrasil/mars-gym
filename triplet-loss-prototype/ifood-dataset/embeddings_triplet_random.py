import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from os.path import expanduser
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

home = expanduser("~")

train_data_dir = '/opt/ifood_dataset/training_data'

ratings_list = []
for file in os.listdir(train_data_dir):
    if file.endswith(".parquet"):
        print(file)
        df = pd.read_parquet(os.path.join(train_data_dir, file))
        ratings_list.append(df)
ratings = pd.concat(ratings_list, axis=0)
print(ratings.shape)
ratings.head(100)

ratings['buys'] = ratings['buys'].fillna(0)
ratings['visits'] = ratings['visits'].fillna(1)

rating_final = ratings[['account_id', 'merchant_id', 'buys']]
repeated_ratings = []
for row in tqdm(rating_final.itertuples(index=False), total=rating_final.shape[0]):
    repeated_ratings.extend([list(row)]*int(row.buys))
rating_final = pd.DataFrame(repeated_ratings, columns=['account_id', 'merchant_id', 'buys'])

del rating_final['buys']
rating_final.head(100)

user_map = pd.DataFrame(rating_final['account_id'].copy())
rating_final['account_id'] = pd.Series(rating_final['account_id']).astype('category').cat.codes
user_map['code'] = rating_final['account_id']
user_map = user_map.drop_duplicates()
print(user_map.head())

rest_map = pd.DataFrame(rating_final['merchant_id'].copy())
rating_final['merchant_id'] = pd.Series(rating_final['merchant_id']).astype('category').cat.codes
rest_map['code'] = rating_final['merchant_id']
rest_map = rest_map.drop_duplicates()
print(rest_map.head())
rating_final.head()

np.savetxt(os.path.join(home, 'models/user_metadata.tsv'), user_map['account_id'].values, delimiter='\t', fmt="%s")
np.savetxt(os.path.join(home, 'models/merchant_metadata.tsv'), rest_map['merchant_id'].values, delimiter='\t', fmt="%s")

users = rating_final['account_id'].unique()
rests = rating_final['merchant_id'].unique()



print(len(users), len(rests))

def generate_triplets(positive_rating):
    triplets = positive_rating.copy()
    triplets.columns = ['user_id', 'pid']
    triplets['nid'] = np.random.choice(rests, len(triplets))
    return triplets['user_id'], triplets['pid'], triplets['nid']


from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, Dot, Lambda
from keras.optimizers import Nadam

BATCH_SIZE = 8192
EPOCHS = 30
LEARNING_RATE = 0.01
EPOCHS_DROP = 3
LATENT_DIM = 64
DROP = 0.9

def plot_figure(list, legend):
	fig = plt.figure(figsize=(6, 6))
	plt.plot(list)
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('step')
	plt.grid()
	plt.legend([legend], loc='upper left')
	fig.savefig(legend + '.png')

def triplet_loss(X):
    pos_dot, neg_dot = X
    m = K.constant(1)
    return neg_dot + K.max(m-pos_dot, 0)

def bpr_triplet_loss(X):
    pos_dot, neg_dot = X
    return 1 - K.sigmoid(pos_dot - neg_dot)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def build_model(num_users, num_items, latent_dim):

    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(
        num_items, latent_dim, name='item_embedding', input_length=1)

    user_input = Input((1, ), name='user_input')

    positive_item_embedding = Flatten(name = "flattened_item_embedding")(item_embedding_layer(
        positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(
        negative_item_input))
    user_embedding = Flatten(name = 'flattened_user_embedding')(Embedding(
        num_users, latent_dim, name='user_embedding', input_length=1)(
            user_input))
    
    pos_dot = Dot(axes=1, normalize = True, name ="pos_dist")([user_embedding, positive_item_embedding])
    neg_dot = Dot(axes=1, normalize = True, name ="neg_dist")([user_embedding, negative_item_embedding])

    loss = Lambda(bpr_triplet_loss)([pos_dot, neg_dot])

    model = Model(
        input=[positive_item_input, negative_item_input, user_input],
        output=loss)
    nadam = Nadam(lr=LEARNING_RATE)
    model.compile(optimizer=nadam,
                  loss=identity_loss)

    return model

latent_dim = LATENT_DIM
model = build_model(len(users), len(rests), latent_dim)
model.summary()


import keras.callbacks 
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []


    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
history = LossHistory()
learning_rate = LEARNING_RATE
loss_log = []
val_loss_log = []
for epoch in range(EPOCHS):

    print('Epoch %s' % epoch)

    if epoch != 0 and epoch % EPOCHS_DROP == 0:
        learning_rate = learning_rate * DROP
        K.set_value(model.optimizer.lr, learning_rate)

    # Sample triplets from the training data
    uid, pid, nid = generate_triplets(rating_final)

    X = {
        'user_input': uid,
        'positive_item_input': pid,
        'negative_item_input': nid
    }

    hist = model.fit(X,
              np.ones(len(uid)),
              batch_size=BATCH_SIZE,
              epochs=1,
              verbose=1,
              shuffle=True,
              callbacks=[history], validation_split=0.1)
    loss_log.append(history.losses)
    val_loss_log.append(hist.history['val_loss'])

model.save(os.path.join(home, 'models/embeddings_deepfood_triplet_randomnegative'))

flat_list = [item for sublist in loss_log for item in sublist]
val_flat_list = [item for sublist in val_loss_log for item in sublist]

plot_figure(flat_list, 'train')
plot_figure(val_flat_list, 'validation')

embedding = Model(input = model.get_layer("user_input").input, output = model.get_layer("flattened_user_embedding").output)
Z = embedding.predict(users)
np.savetxt(os.path.join(home, "models/user_embeddings.tsv"), Z , delimiter="\t")

embedding = Model(input = model.get_layer("positive_item_input").input, output = model.get_layer("flattened_item_embedding").output)
Z = embedding.predict(rests)
np.savetxt(os.path.join(home, "models/restaurant_embeddings.tsv"), Z , delimiter="\t")