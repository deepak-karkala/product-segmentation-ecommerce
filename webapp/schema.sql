DROP TABLE IF EXISTS product;
CREATE TABLE product (
id INTEGER PRIMARY KEY AUTOINCREMENT,
category TEXT NOT NULL,
image_path TEXT NOT NULL
);
INSERT INTO product(category, image_path)
VALUES
('Bed', '168.jpg'),
('Chair', '086.jpg');