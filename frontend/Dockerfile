FROM node:lts-alpine AS base
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .

FROM base AS dev
EXPOSE 80
CMD ["npm", "run", "dev"]

FROM base AS build
RUN npm run build

FROM joseluisq/static-web-server:latest AS prod
COPY --from=build /app/dist /app
EXPOSE 80
CMD ["static-web-server", "--port", "80", "--root", "/app"]