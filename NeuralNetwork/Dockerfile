FROM microsoft/dotnet:2.0-runtime-nanoserver-1709 AS base
WORKDIR /app

FROM microsoft/dotnet:2.0-sdk-nanoserver-1709 AS build
WORKDIR /src
COPY *.sln ./
COPY NeuralNetwork/NeuralNetwork.csproj NeuralNetwork/
RUN dotnet restore
COPY . .
WORKDIR /src/NeuralNetwork
RUN dotnet build -c Release -o /app

FROM build AS publish
RUN dotnet publish -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "NeuralNetwork.dll"]
