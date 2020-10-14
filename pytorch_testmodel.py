# start testing

data_file = pd.read_csv('/af2020cv-2020-05-09-v5-dev/test.csv')
File_id = data_file["FileID"].values.tolist()

for i in range(len(File_id)):
    test_dir = File_id[i] + '.jpg'
    img_dir = '/image/test/' + test_dir
    # load image
    img = Image.open(img_dir)
    inputs = data_transforms(img)
    inputs.unsqueeze_(0)

    if use_gpu:
        model = model_ft.cuda()  # use GPU
    else:
        model = model_ft
    model.eval()
    if use_gpu:
        inputs = Variable(inputs.cuda())  # use GPU
    else:
        nputs = Variable(inputs)

    # forward
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    class_name = get_key(mapping, preds.item())
    class_name = '%s' % (class_name)
    class_name = class_name[2:-2]

    print(img_dir)
    print('prediction_label:', class_name)
    print(30 * '--')
    Species_id.append(class_name)

test = pd.DataFrame({'FileId': File_id, 'SpeciesID': Species_id})  # 将机器分类结果存储在.csv文件中
test.to_csv('result.csv', index=None, encoding='utf8')
