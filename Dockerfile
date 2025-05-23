# Utiliser une image de base avec Python 3.10 et CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    wget \
    git \
    && apt-get clean

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . /app

# Configurer pip pour Python 3.10
RUN python3.10 -m pip install --upgrade pip

# Installer les dépendances Python
RUN python3.10 -m pip install -r requirements.txt

# Exposer le port 5000 pour l'API
EXPOSE 5000

# Commande par défaut pour démarrer le serveur
CMD ["python3.10", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]