
## 1. Giới thiệu

- Đoạn code này thực hiện xây dựng và huấn luyện mô hình mạng nơ-ron tích chập (CNN) để phân loại bệnh trên lá cây từ bộ dữ liệu PlantVillage. Mục tiêu là đạt độ chính xác trên tập validation trên 90%.

## 2.1. Cài đặt thư viện và kiểm tra thiết bị
```py
import torch
import torch.nn as nn
...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- Sử dụng PyTorch, torchvision, matplotlib, scikit-learn.

- Tự động phát hiện GPU nếu có, nếu không sẽ dùng CPU.

## 2.2. Tải dữ liệu từ kaggle
```py
import kagglehub
path = kagglehub.dataset_download("emmarex/plantdisease")
```
- Sử dụng thư viện kagglehub để tải bộ dữ liệu PlantVillage từ Kaggle.

- Đường dẫn dữ liệu được lưu vào biến path.

## 2.3. Kiểm tra và phân tích dữ liệu
- Liệt kê các lớp (class) và số lượng ảnh trong mỗi lớp.

- Vẽ biểu đồ phân phối dữ liệu.

- Nhận xét: dữ liệu bị mất cân bằng (class `Potato___healthy` chỉ có 152 ảnh, trong khi class `Tomato__Tomato_YellowLeaf__Curl_Virus` có 3208 ảnh).

## 2.4. Xử lý mất cân bằng dữ liệu

```py
train_sampler = WeightedRandomSampler(weights=train_samples_weight, num_samples=len(train_indices), replacement=True)
```

- Sử dụng `WeightedRandomSampler` để tạo sampler có trọng số.

- Mỗi mẫu được gán trọng số tỉ lệ nghịch với số lượng ảnh trong lớp của nó.

- Giúp mô hình học tốt hơn trên các lớp ít dữ liệu.

## 2.5. Tăng cường dữ liệu (Data Augmentation)
```py
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(...),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```
- Áp dụng các phép biến đổi ngẫu nhiên để tăng tính đa dạng dữ liệu.

- Chuẩn hóa ảnh về giá trị trung bình và độ lệch chuẩn của ImageNet.

## 2.6. Xây dựng mô hình CNN

- Mô hình PlantCNN gồm:

    - 4 block tích chập: mỗi block gồm 2 lớp Conv2D + BatchNorm + ReLU + MaxPooling + Dropout2D.

    - 3 lớp fully connected sau khi flatten.

    - Dropout mạnh (0.5) để tránh overfitting.

    - Tổng số tham số: 9.7 triệu.

## 2.7. Huấn luyện mô hình
```py
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

- Loss: CrossEntropyLoss.

- Optimizer: Adam với weight decay.

- Scheduler: giảm learning rate nếu loss không giảm sau 5 epoch.

Vòng lặp huấn luyện:

- Mỗi epoch: train → validation.

- Lưu lại model có validation accuracy cao nhất.

- Dừng sớm nếu đạt target 90% hoặc không cải thiện sau 5 epoch.

## 2.8. Kết quả
- Sau 14 epoch, mô hình đạt validation accuracy 90.99%.

- Lưu model tốt nhất: best_plant_model.pth.

- Vẽ biểu đồ accuracy và loss theo epoch.