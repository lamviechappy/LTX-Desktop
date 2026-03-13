curl -X POST http://localhost:5173/api/models/download \
  -H "Content-Type: application/json" \
  -d '{"skipTextEncoder": false}'


  curl http://localhost:5173/api/models/download/progress


  curl http://localhost:5173/api/models/status