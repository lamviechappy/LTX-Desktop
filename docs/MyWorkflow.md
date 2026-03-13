# WORKFLOW
Để quản lý tốt việc vừa cập nhật từ bản gốc (upstream), vừa phát triển tính năng chạy local trên Mac (M4), bạn nên thiết lập luồng công việc (workflow) theo các bước sau:
1. Thiết lập Remote Upstream
Hiện tại bản clone trên máy chỉ kết nối với bản fork của bạn (origin). Bạn cần kết nối thêm với bản gốc của Lightricks để lấy cập nhật mới nhất.
bash
# Thêm remote từ bản gốc
git remote add upstream https://github.com/Lightricks/LTX-Desktop

# Kiểm tra lại (sẽ thấy cả origin và upstream)
git remote -v
Hãy thận trọng khi sử dụng mã.

2. Luồng đồng bộ nhánh main
Nhánh main trên máy bạn nên dùng để soi chiếu với bản gốc. Đừng code trực tiếp trên main.
Để cập nhật main từ bản gốc:
bash
git checkout main
git pull upstream main
git push origin main
Hãy thận trọng khi sử dụng mã.

3. Tạo nhánh mới để phát triển tính năng Mac Local
Đây là nơi bạn sẽ thực hiện các chỉnh sửa để hỗ trợ model LTX chạy locally trên chip M4.
bash
# Tạo nhánh mới tên là 'feat/mac-local-support'
git checkout -b feat/mac-local-support
Hãy thận trọng khi sử dụng mã.

4. Quy trình làm việc hàng ngày (Workflow)
Code & Commit: Làm việc trên nhánh feat/mac-local-support.
bash
git add .
git commit -m "feat: setup basic MPS support for local inference"
Hãy thận trọng khi sử dụng mã.

Đẩy lên Github của bạn:
bash
git push origin feat/mac-local-support
Hãy thận trọng khi sử dụng mã.

Cập nhật code mới từ bản gốc (nếu bản gốc có update):
bash
git checkout main
git pull upstream main
git checkout feat/mac-local-support
git merge main
Hãy thận trọng khi sử dụng mã.

5. Lưu ý cho dự án LTX-Desktop trên Mac M4
Vì bạn đang muốn làm cho app hỗ trợ local inference trên Mac:
File cần chú ý: Hãy tìm các file liên quan đến cấu hình backend (thường là Python). Kiểm tra cách họ gọi torch.device("cuda") và sửa/bổ sung để hỗ trợ torch.device("mps").
Dependencies: Đảm bảo uv đã cài đặt đúng phiên bản torch hỗ trợ Silicon (bước trước bạn thấy MPS available: True là tín hiệu rất tốt).
File docs/Notes.md: Hiện đang là untracked. Bạn có thể thêm nó vào .gitignore nếu đó là ghi chú cá nhân, hoặc git add nếu muốn lưu nó lên repo để theo dõi tiến độ.