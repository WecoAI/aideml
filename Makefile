.PHONY: docker docker-build docker-run install clean

# Docker image name
IMAGE_NAME = aide

# Python version and venv
PYTHON = python3.10
VENV_NAME = .venv

# Default directories for logs and workspaces
WORKSPACE_BASE ?= $(PWD)/workspaces
LOGS_DIR ?= $(PWD)/logs

# Virtual environment installation
install:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_NAME)
	@echo "Installing dependencies..."
	@. $(VENV_NAME)/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt && \
		pip install -e .
	@echo "Installation complete. Activate the virtual environment with: source $(VENV_NAME)/bin/activate"

# Docker commands combined
docker: docker-build docker-run

# Build Docker image
docker-build:
	docker build -t $(IMAGE_NAME) .

# Run Docker container
docker-run:
	@mkdir -p "$(LOGS_DIR)" "$(WORKSPACE_BASE)"
	docker run -it --rm \
		-v "$(LOGS_DIR):/app/logs" \
		-v "$(WORKSPACE_BASE):/app/workspaces" \
		-v "$(PWD)/aide/example_tasks:/app/data" \
		-e OPENAI_API_KEY="$(OPENAI_API_KEY)" \
		$(IMAGE_NAME) \
		data_dir=/app/data/house_prices \
		goal="Predict the sales price for each house" \
		eval="Use the RMSE metric between the logarithm of the predicted and observed values."

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	rm -rf workspaces/* logs/*
	docker rmi $(IMAGE_NAME) || true
