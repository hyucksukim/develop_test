FROM node:17.9.1-buster

WORKDIR /app

COPY ./*.json /app/

COPY . /app/

RUN npm install

EXPOSE 3000

CMD ["npm", "start"]

