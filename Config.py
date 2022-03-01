# data preprocess
data_path = 'dataset'
train_file = 'train_pinggu.csv'
valid_file = 'test_pinggu.csv'
test_file = 'test_pinggu.csv'
fix_length = 256
batch_size = 256
# data label list
label_list = ['低危', '中危', '高危', '超危']
class_number = len(label_list)
# train details
epochs = 30
learning_rate = 1e-3