# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

def findNextCellToFill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 *int(i/3), 3 *int(j/3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
                return True
    return False

def solveSudoku(grid, i=0, j=0):
    i,j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid,i,j,e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False

## Model Trainning KNN
samples = np.load('samples.npy')
labels = np.load('label.npy')

k = 80
train_label = labels[:k]
train_input = samples[:k]
test_input = samples[k:]
test_label = labels[k:]

model = cv2.ml.KNearest_create()
model.train(train_input,cv2.ml.ROW_SAMPLE,train_label)

#retval, results, neigh_resp, dists = model.findNearest(test_input, 1)
#string = results.ravel()
#print(string)
#print(test_label.reshape(1,len(test_label))[0])

img = cv2.imread('./images/002.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Phân đoạn ngưỡng
## 0: Binary
#1: Binary Inverted
#2: Threshold Truncated
#3: Threshold to Zero
#4: Threshold to Zero Inverted

# THRESH_BINARY
#     Nếu giá trị pixel lớn hơn ngưỡng thì gán bằng maxval
#     Ngược lại bằng gán bằng 0
# THRESH_BINARY_INV
#     Nếu giá trị pixel lớn hơn ngưỡng thì gán bằng 0
#     Ngược lại bằng gán bằng maxval
# THRESH_TRUNC
#     Nếu giá trị pixel lớn hơn ngưỡng thì gán giá trị bằng ngưỡng
#     Ngược lại giữ nguyên giá trị
# THRESH_TOZERO
#     Nếu giá trị pixel lớn hơn ngưỡng thì giữ nguyên giá trị
#     Ngược lại gán bằng 0
# THRESH_TOZERO_INV
#     Nếu giá trị pixel lớn hơn ngưỡng thì gán giá trị bằng 0
#     Ngược lại giữ nguyên

    
ret,thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# plt.subplot(121), plt.imshow(img, cmap='Greys_r')
# plt.subplot(122), plt.imshow(thresh, cmap='Greys_r')
# plt.show()

# print("ret: %s, thresh: %s" %(ret, thresh))
# # Rectangular Kernel
# >>> cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# array([[1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1]], dtype=uint8)

# # Elliptical Kernel
# >>> cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# array([[0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0]], dtype=uint8)

# # Cross-shaped Kernel
# >>> cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# array([[0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0]], dtype=uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))   
dilated = cv2.dilate(thresh,kernel)

# print(dilated);
# cv2.namedWindow("img", cv2.WINDOW_NORMAL);
# cv2.imshow("img", dilated)
 
## Đường viền
#Contours là đường bao kết nối tất cả các điểm liền kề nhau có cùng màu sắc hoặc độ tương phản.
#Chính vì đặc tính này, contours thường được dùng trong xác định vật thể, nhận dạng,... 
image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print("h:%s" %hierarchy)
# for cnt in contours:
#     cv2.drawContours(img, [cnt], -1, 255, -1)
# plt.subplot(121), plt.imshow(img)
# plt.subplot(122), plt.imshow(img)
# plt.show()

## Giải nén tám mươi mốt ô vuông nhỏ
boxes = []
for i in range(len(hierarchy[0])):
    # print("hierarchy: %s" %hierarchy[0][i][3])
    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])
# print("boxes", boxes)
# print("shape",img.shape);
height,width = img.shape[:2]
box_h = height/9 # chiều cao của ô
box_w = width/9 # chiều rộng của ô
number_boxes = []
##Sudoku được khởi tạo thành mảng giá trị 0
soduko = np.zeros((9, 9),np.int32)
haveNumber = []
#[Next, Previous, First_Child, Parent]
for j in range(len(boxes)):
    if boxes[j][2] != -1: # có con
        #number_boxes.append(boxes[j])
        x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])
        haveNumber.append([x,y])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #Vẽ khung bao quanh số
        number_boxes.append([x,y,w,h])
        #img = cv2.rectangle(img,(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)
        #img = cv2.drawContours(img, contours, boxes[j][2], (0,255,0), 1)
        ## Xử lý các số được trích xuất
        number_roi = gray[y:y+h, x:x+w]
        # print("number_roi %s" %number_roi)
        ## Kích thước đồng nhất
        #ADAPTIVE_THRESH_MEAN_C: giá trị của pixel phụ thuộc vào các pixel lân cận
        #ADAPTIVE_THRESH_GAUSSIAN_C: giá trị của pixel cũng phụ thuộc vào các pixel lân cận, tuy nhiên được khử nhiễu
        resized_roi=cv2.resize(number_roi,(20,40))
        thresh1 = cv2.adaptiveThreshold(resized_roi,255,1,1,11,2) 
        ## Giá trị pixel được chuẩn hóa
        normalized_roi = thresh1/255.  
        
        ## Mở rộng thành hàng để cho phép xác định knn
        sample1 = normalized_roi.reshape((1,800))
        # print("truoc:",sample1)
        sample1 = np.array(sample1,np.float32)
        # print("sau:",sample1)
        ## Nhận dạng knn
        retval, results, neigh_resp, dists = model.findNearest(sample1, 1)
        # print(results)  
        # print(results.ravel()) 
        number = int(results.ravel()[0])
        # print("number: %s" %number)
        ## Hiển thị kết quả nhận dạng (số màu xanh dương)
        cv2.putText(img,str(number),(x+w,y+h), 3, 2., (255, 0, 0), 2, cv2.LINE_AA)
        
        ## Tìm vị trí trong ma trận
        soduko[int(y/box_h)][int(x/box_w)] = number
               
        #print(number)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL); 
        cv2.imshow("img", img)
        cv2.waitKey(30)
print("\nSudoku đã tạo\n")
print(soduko)
print("\nSudoku sau khi giải quyết\n")

## Giai phap cua sodoku
solveSudoku(soduko)

print(soduko)
print("\nKiểm tra: Tìm tổng của mỗi hàng và cột\n")
row_sum = map(sum,soduko)
col_sum = map(sum,zip(*soduko))
print(list(row_sum))
print(list(col_sum))
print("haveNumber len:", len(haveNumber))
print("haveNumber:", haveNumber)
#print(sum(soduko.transpose))
## Đặt kết quả vào hình ảnh vào vị trí  
for i in range(9):
    for j in range(9):
        x = int((i+0.25)*box_w)
        y = int((j+0.5)*box_h)
        print("pos: %s" %[x,y])
        cv2.putText(img,str(soduko[j][i]),(x,y), 3, 2.5, (0, 0, 255), 2, cv2.LINE_AA)
#print(number_boxes)
cv2.namedWindow("img", cv2.WINDOW_NORMAL);
cv2.imshow("img", img)
cv2.waitKey(0)


#retval, results, neigh_resp, dists = model.findNearest(test_input, 1)
#string = results.ravel()
#print(string)
#print(test_label.reshape(1,len(test_label))[0])

'''
C = 5
gamma = 0.5
model = cv2.ml.SVM_create()
model.setGamma(gamma)
model.setC(C)
model.setKernel(cv2.ml.SVM_LINEAR)
model.setType(cv2.ml.SVM_C_SVC)
model.train(train_input,cv2.ml.ROW_SAMPLE,train_label)
predict_label = model.predict(test_input)[1].ravel()
print(predict_label)
print(test_label.reshape(1,len(test_label))[0])


class MLP():
    class_n = 10
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()

    def unroll_responses(self, responses):
        sample_n = len(responses)
        labels = []
        for i in range(len(responses)):
            label_b = np.zeros(self.class_n, np.int32)
            label = responses[i]
            label_b[int(label)] = 1
            labels.append(label_b)
        #print(labels)
            
        return labels
    
    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses)
        layer_sizes = np.int32([var_n, 200, 50, self.class_n])

        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0.1)
        self.model.setBackpropWeightScale(0.001)
        #self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.1))
        self.model.setTermCriteria((cv2.TERM_CRITERIA_EPS, 50, 0.001))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

model = MLP()
model.train(train_input,train_label)
predict_label = model.predict(test_input)
print(predict_label)
print(test_label.reshape(1,len(test_label))[0])
 
''' 