FROM node:18-alpine
WORKDIR /app
COPY frontend/package.json /app/package.json
RUN npm install --legacy-peer-deps || npm install
COPY frontend /app
EXPOSE 3000
CMD ["npm", "run", "dev"]

