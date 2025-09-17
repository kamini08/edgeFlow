# EdgeFlow Docker Documentation

## Quick Start
```
# Build all images
make docker-build

# Start all services
make docker-up

# Check service status
docker-compose ps

# View logs
make docker-logs

# Stop services
make docker-down
```

## Running EdgeFlow in Docker

### Single Optimization
```
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/configs:/app/configs \
           -v $(pwd)/outputs:/app/outputs \
           edgeflow:latest config.ef
```

### With Custom Device Specs
```
docker run -v $(pwd)/device_specs:/app/device_specs \
           edgeflow:latest config.ef --device-spec-file /app/device_specs/custom.csv
```

## Development

### Building Images
```
# Build without cache
docker-compose build --no-cache

# Build specific service
docker-compose build edgeflow-api
```

### Debugging
```
# Enter container shell
docker exec -it edgeflow-compiler sh

# View specific service logs
docker-compose logs -f edgeflow-api
```

## Production Deployment

### Using Docker Swarm
```
docker stack deploy -c docker-compose.prod.yml edgeflow
```

## Environment Variables

- LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
- API_KEY: API authentication key
- CORS_ORIGINS: Allowed CORS origins for API

## Troubleshooting

### Container won't start
```
# Check logs
docker logs edgeflow-compiler

# Verify image built correctly
docker images | grep edgeflow
```

### Permission issues
```
# Fix volume permissions
sudo chown -R $USER:$USER ./models ./outputs
```

