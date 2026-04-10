# Spoof Check

Spoof Check is a service for detecting spoofing in audio files. It uses a deep learning model to analyze the audio and determine if it is genuine or spoofed.

## 🛠️ Getting Started

Follow the steps below to set up and run the service using Docker

### ⚙️ Configure Environment Variables

Copy the example environment file and fill in the necessary values:

```bash
cp .env.example .env
```

Edit the `.env` file to set your environment variables. You can use the default values or customize
them as needed.

Also, make sure to configure the `docker-compose.yml` file if necessary.

### 🐳 Build and Run the Docker Container

Start the Docker container with the following commands:

```bash
docker compose build
```

```bash
docker compose up -d
```

This command will build the Docker image and start the container.
