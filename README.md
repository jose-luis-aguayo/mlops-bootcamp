## 1. Build a docker image
```
docker build -t sales_forecast .
```

## 1.Start a container
```
docker run -d --name sales_forecast_container -p 80:80 sales_forecast 
```