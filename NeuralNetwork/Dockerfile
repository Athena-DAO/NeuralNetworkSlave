FROM microsoft/dotnet:2.0-runtime AS base
WORKDIR /app

FROM microsoft/dotnet:2.0-sdk AS build
WORKDIR /src
COPY NeuralNetwork/NeuralNetwork.csproj NeuralNetwork/
RUN dotnet restore NeuralNetwork/NeuralNetwork.csproj
COPY . .
WORKDIR /src/NeuralNetwork
RUN dotnet build NeuralNetwork.csproj -c Release -o /app

FROM build AS publish
RUN dotnet publish NeuralNetwork.csproj -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
COPY NeuralNetwork/appsettings.json /app
ENTRYPOINT ["dotnet", "NeuralNetwork.dll"]
