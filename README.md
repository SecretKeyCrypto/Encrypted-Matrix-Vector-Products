## 🔧 Installation

### 🛠 Prerequisites

Make sure you have the following installed:

- **Go ≥ 1.21**
- **C++17** compiler (`g++`)

Install on Ubuntu/Debian:

```
sudo apt update
sudo apt install build-essential
```

---

### ⚙️ Step 1: Compile Native C++ Libraries

Run the setup script to compile the required C++ components:

```
./run.sh
```

This generates the following static libraries:

- `tdm/libNTT.a`
- `ecc/libReedSolomon.a`
- `mvp/libMVP.a`

**Note:** You must run this command from the **root** of the repository.

---

### 🌐 Step 2: Export Paths for Go to Use Native Libraries

Before running tests, set the appropriate environment variables **from the root directory** of this project:

```
export CGO_CXXFLAGS="-std=c++17 -I$(pwd)/tdm -I$(pwd)/ecc -I/usr/include"
export CGO_LDFLAGS="-L$(pwd)/tdm -L$(pwd)/ecc -lNTT -lReedSolomon"
```

---

### 🚀 Step 3: Run Go Benchmarks

From the project root (or inside the `mvp/` folder if that's where your `go.mod` is), run:

```
go test -bench=. ./...
```

This will compile the Go code, link against the native C++ libraries, and execute the performance benchmarks.

---


### 🍏 macOS Notes (Apple Silicon / Intel Mac)

On macOS, OpenSSL is not provided by default. This project requires **libcrypto** (part of OpenSSL).

1. Install OpenSSL via Homebrew:
   ```bash
   brew install openssl@3
   ```

2. Export flags so Go/cgo can find it:
   ```bash
   export CGO_CFLAGS="-I$(brew --prefix openssl@3)/include $CGO_CFLAGS"
   export CGO_LDFLAGS="-L$(brew --prefix openssl@3)/lib $CGO_LDFLAGS"
   ```

   > 💡 Tip: these variables are already set automatically if you run `./run.sh` from the project root.

---

### 🛠 Example full setup on macOS

```bash
brew install openssl@3
./run.sh
export CGO_CFLAGS="-I$(brew --prefix openssl@3)/include -I$(pwd)/tdm -I$(pwd)/ecc"
export CGO_LDFLAGS="-L$(brew --prefix openssl@3)/lib -L$(pwd)/tdm -L$(pwd)/ecc -lNTT -lReedSolomon -lcrypto"
go test -bench=. ./...
```