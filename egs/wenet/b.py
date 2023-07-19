import h5py
f = h5py.File("data/tokenized/wenet_encodec_train_m.h5","r")

print(f.keys())
