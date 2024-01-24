import pandas as pd
import numpy as np
import collections
import scipy.sparse as sp
import networkx as nx

# data = pd.read_csv("data/ml-100k/u.data", '\t', header=None, names=['userid', 'itemid', 'rating', 'time'],
#                    encoding='utf-8')
# # print(data.head(10))
#
# user = pd.read_csv("data/ml-100k/u.user", '|', header=None, names=['userid', 'age', 'gender', 'type', 'id'])
# user['gender1'] = user['gender'].map({'M': 0, 'F': 1})
#
# with open('data/ml-100k/u.occupation', 'r') as f:
#     lines = f.readlines()
#     index = 0
#     occup = {}
#     for i in range(len(lines)):
#         occup[lines[i][:-1]] = i
# print(occup)
# user['type1'] = user['type'].map(occup)
#
# print(data.head(10))
# max_user = np.max(np.array(data['userid']))
# max_item = np.max(np.array(data['itemid']))
# a = np.zeros((max_user, max_item))
# for index, row in data.iterrows():
#     u, i, r, _ = row
#     a[u - 1, i - 1] = r / 5
# np.save("data/ml-100k/data_train.npy", a[:-1])
# np.save("data/ml-100k/data_test.npy", a[-1:])
# age = np.array(user['age'])
# age = (age - np.min(age)) / (np.max(age) - np.min(age))
# np.save("data/ml-100k/age_train.npy", age[:-1, None])
# np.save("data/ml-100k/age_test.npy", age[-1:, None])
# gender = np.array(user['gender1'])
# gender = np.eye(2)[gender]
# np.save("data/ml-100k/gender_train.npy", gender[:-1])
# np.save("data/ml-100k/gender_test.npy", gender[-1:])
# type = np.array(user['type1'])
# type = np.eye(21)[type]
# np.save('data/ml-100k/type_train.npy', type[:-1])
# np.save('data/ml-100k/type_test.npy', type[-1:])

# import scipy.sparse as sp
#
# sns = sp.load_npz("data/weibo/sns.npz")
# user = sp.load_npz('data/weibo/user.npz')
# feature = user[:, [0, 1]].A
# index = np.where((feature[:, 0] > 0) * (feature[:, 0] < 100) * (feature[:, 1] > 0) * (feature[:, 1] < 3))[0]
# print(len(index))
# user = user[list(index)]
# sns = sns[list(index)][:, list(index)]
# print(user.shape, sns.shape)
# # sp.save_npz('data/weibo/sns_filter.npz', sns)
# # sp.save_npz('data/weibo/user_filter.npz', user)
#
# count1 = np.zeros(sns.shape[0])
# count2 = np.zeros(sns.shape[0])
# sns_temp = sns.tocoo()
# row = sns_temp.row
# col = sns_temp.col
# for i in range(len(row)):
#     if (i % 10000 == 0):
#         print(i)
#     count1[row[i]] += 1
#     count2[col[i]] += 1
# count1 = np.array(count1)
# count2 = np.array(count2)
# index = np.where((count2 > 5) * (count1) > 5)[0]
# print(len(index))
# user = user[list(index)]
# sns = sns[list(index)][:, list(index)]
# print(user.shape, sns.shape)
# sp.save_npz('data/weibo/sns_filter.npz', sns)
# sp.save_npz('data/weibo/user_filter.npz', user)

# user_id_map = collections.defaultdict(lambda: -1)
# count = 0
# tag_count = [0] * 1000000
# with open("data/weibo/track/user_profile.txt", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         userid, birth, gender, tweets, tagid = line.strip().split("\t")
#         if (int(gender) == 0 or int(gender) > 2 or '-' in birth or tagid=='0'):
#             continue
#         else:
#             user_id_map[userid] = count
#             count += 1
#             tagids = map(int, tagid.split(";"))
#             for ids in tagids:
#                 tag_count[ids] += 1
# tag_count = np.argsort(-np.array(tag_count))[:2000]
# tag_idx_map = collections.defaultdict(lambda: -1)
# for i in range(len(tag_count)):
#     tag_idx_map[tag_count[i]] = i
#
# all_data = []
# all_label = []
# with open("data/weibo/track/rec_log_train.txt", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         user1, user2, label, _ = line.strip().split()
#         if (user_id_map[user1] >= 0 and user_id_map[user2] >= 0):
#             all_data.append([user_id_map[user1], user_id_map[user2]])
#             all_label.append((int(label) + 1) // 2)
# all_data = np.array(all_data)
# print(all_data.shape)
# all_label = np.array(all_label)
# np.save("data/weibo/all_data.npy", all_data)
# np.save("data/weibo/all_label.npy", all_label)
#
# key_count = [0] * 1000000
# with open("data/weibo/track/user_key_word.txt", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         userid, keywords = line.strip().split("\t")
#         keywords = keywords.split(";")
#         for keyword in keywords:
#             key, weight = keyword.split(':')
#             key_count[int(key)] += 1
# key_count = np.argsort(-np.array(key_count))[:2000]
# key_idx_map = collections.defaultdict(lambda: -1)
# for i in range(len(key_count)):
#     key_idx_map[key_count[i]] = i
#
# cols = []
# rows = []
# values = []
# labels = []
# with open("data/weibo/track/user_profile.txt", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         userid, birth, gender, tweets, tagid = line.strip().split("\t")
#         if (int(gender) == 0 or int(gender) > 2 or '-' in birth or tagid=='0'):
#             continue
#         else:
#             labels.append(int(gender) - 1)
#             row = user_id_map[userid]
#             tagids = map(int, tagid.split(";"))
#             for ids in tagids:
#                 if (tag_idx_map[ids] >= 0):
#                     rows.append(row)
#                     cols.append(tag_idx_map[ids])
#                     values.append(1)
#             rows.append(row)
#             cols.append(4000)
#             values.append(2023 - float(birth))
#             rows.append(row)
#             cols.append(4001)
#             values.append(float(tweets))
# with open("data/weibo/track/user_key_word.txt", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         userid, keywords = line.strip().split("\t")
#         if (user_id_map[userid] >= 0):
#             keywords = keywords.split(";")
#             for keyword in keywords:
#                 key, weight = keyword.split(':')
#                 if (key_idx_map[int(key)] >= 0):
#                     rows.append(user_id_map[userid])
#                     cols.append(2000 + key_idx_map[int(key)])
#                     values.append(float(weight))
#
# data = sp.coo_matrix((values, (rows, cols)), shape=[count, 4002])
# sp.save_npz("data/weibo/feature.npz", data)
# np.save("data/weibo/secret_label.npy", np.eye(2)[np.array(labels)])
#
# assert len(labels) == count
# print(count)
#
# G = nx.Graph()
# G.add_nodes_from(list(np.arange(count)))
#
# cols = []
# rows = []
# values = []
# edges = []
# with open("data/weibo/track/user_sns.txt", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         user1, user2 = line.strip().split('\t')
#         if (user_id_map[user1] >= 0 and user_id_map[user2] >= 0):
#             rows.append(user_id_map[user1])
#             cols.append(user_id_map[user2])
#             values.append(1)
#             edges.append((user_id_map[user1], user_id_map[user2]))
# adj = sp.coo_matrix((values, (rows, cols)), shape=[count, count])
# sp.save_npz("data/weibo/adj.npz", adj)

adj = sp.load_npz("data/weibo/adj.npz").tocsr()
features = sp.load_npz('data/weibo/feature.npz').tocsr()
secret_label = np.load("data/weibo/secret_label.npy")

all_data = np.load("data/weibo/all_data.npy")
all_label = np.load("data/weibo/all_label.npy")

order = np.arange(features.shape[0])

while (True):
    followers = np.array(adj.sum(0))[0]
    print(followers.shape)
    idx = np.where(followers < 2)[0]
    if (len(idx) == 0):
        break
    idx = np.where(followers >= 2)[0]
    adj = adj[idx][:, idx]
    features = features[idx]
    secret_label = secret_label[idx]
    order = order[idx]

maps = collections.defaultdict(lambda: -1)
for i in range(len(order)):
    maps[order[i]] = i

all_data_list = []
all_label_list = []
for i in range(len(all_data)):
    if (maps[all_data[i, 0]] >= 0 and maps[all_data[i, 1]] >= 0):
        all_data_list.append([maps[all_data[i, 0]], maps[all_data[i, 1]]])
        all_label_list.append(all_label[i])

all_data = np.array(all_data_list)
all_label = np.array(all_label_list)
print(np.min(all_data), np.max(all_data))
np.save("data/weibo/filter_all_data.npy", all_data)
np.save("data/weibo/filter_all_label.npy", all_label)

print(adj.shape)
print(features.shape)

features = features.toarray()
features_temp = features[:, :2000]
features_temp[features_temp == 2] = 1
features[:, :2000] = features_temp
features_temp = features[:, 2000:4000]
features_temp = (features_temp - np.min(features_temp)) / (np.max(features_temp) - np.min(features_temp))
features[:, 2000:4000] = features_temp
features[:, -2:] = (features[:, -2:] - np.min(features[:, -2:], axis=0, keepdims=True)) / (
        np.max(features[:, -2:], axis=0, keepdims=True) - np.min(features[:, -2:], axis=0, keepdims=True))
features = sp.csr_matrix(features)

sp.save_npz("data/weibo/filter_adj.npz", adj)
sp.save_npz("data/weibo/filter_feature.npz", features)
np.save("data/weibo/filter_secret_label.npy", secret_label)
