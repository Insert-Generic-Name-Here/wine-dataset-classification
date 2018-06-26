CREATE TABLE vinho_verde_red(
	fixed_acidity real,
	volatile_acidity real,
    citric_acid real,
    residual_sugar real,
    chlorides real,
    free_sulfur_dioxide real,
    total_sulfur_dioxide real,
    density real,
    pH real,
    sulphates real,
    alcohol real,
 	quality int
);

CREATE TABLE vinho_verde_white(
	fixed_acidity real,
	volatile_acidity real,
    citric_acid real,
    residual_sugar real,
    chlorides real,
    free_sulfur_dioxide real,
    total_sulfur_dioxide real,
    density real,
    pH real,
    sulphates real,
    alcohol real,
 	quality int
);

COPY vinho_verde_red 
FROM 'C:\Users\This PC\Documents\Coding\Python\wine-dataset-classification\winequality-red.csv'
WITH (FORMAT csv, DELIMITER ';', HEADER true);

COPY vinho_verde_white 
FROM 'C:\Users\This PC\Documents\Coding\Python\wine-dataset-classification\winequality-white.csv' 
WITH (FORMAT csv, DELIMITER ';', HEADER true);