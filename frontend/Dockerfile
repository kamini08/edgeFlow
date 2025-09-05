FROM node:18-alpine
WORKDIR /app
COPY package.json /app/package.json
RUN npm install --legacy-peer-deps || npm install
COPY . /app
EXPOSE 3000
CMD ["npm", "run", "dev"]
