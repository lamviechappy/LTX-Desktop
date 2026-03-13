# Hướng dẫn cài đặt

1. Cần có
Node.js
uv (Python package manager)
Python 3.12+
Git

2. Tạo môi trường Python 3.13 bằng Conda
Thay vì để script tự tạo venv bằng uv, hãy tạo môi trường bằng Conda trước:
bash
conda create -n ltx python=3.13.12
conda activate ltx

3. "Mách" cho uv biết Python của Conda nằm ở đâu
Vì script đang dùng uv, bạn cần trỏ uv vào đúng Python mà Conda vừa tạo để nó không đi tìm ở nơi khác nữa:

export UV_PYTHON=$(which python)
pnpm setup:dev:mac

4. Run:

pnpm dev
Debug:

pnpm dev:debug
dev:debug starts Electron with inspector enabled and starts the Python backend with debugpy.

Typecheck:

pnpm typecheck
Backend tests:

pnpm backend:test
Building installers:

See INSTALLER.md