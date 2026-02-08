---
title: SuperKart Sales Predictor
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SuperKart Sales Predictor (Docker Space)

This Space exposes a FastAPI endpoint.

## Endpoints
- `GET /` health check
- `POST /predict` returns predicted sales

## Example JSON for /predict

    {
      "Product_Weight": 12.5,
      "Product_Sugar_Content": "Low Sugar",
      "Product_Allocated_Area": 18.0,
      "Product_Type": "Fruits and Vegetables",
      "Product_MRP": 249.0,
      "Store_Size": "Medium",
      "Store_Location_City_Type": "Tier 2",
      "Store_Type": "Supermarket Type1",
      "Store_Age": 10
    }
