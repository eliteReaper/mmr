import 'package:flutter/material.dart';

// ignore: non_constant_identifier_names
Widget ItemCard(
    String name, String rating, String genre, BuildContext context) {
  return InkWell(
    onTap: () {},
    child: Card(
      elevation: 5,
      child: SizedBox(
        child: Column(
          children: [
            Container(
              constraints: const BoxConstraints(maxWidth: 300),
              child: Text(
                name,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 20,
                ),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(
              height: 5,
            ),
            Container(
              constraints: const BoxConstraints(maxWidth: 300),
              padding: const EdgeInsets.all(5),
              child: Text(
                genre,
                style: const TextStyle(
                  color: Colors.black,
                  fontSize: 16,
                ),
                textAlign: TextAlign.center,
              ),
              decoration: BoxDecoration(
                border: Border.all(
                  width: 1,
                  color: Colors.black38,
                  style: BorderStyle.solid,
                ),
                borderRadius: const BorderRadius.all(
                  Radius.circular(15),
                ),
                color: Colors.grey.shade300,
              ),
              // color: ,
            ),
            const SizedBox(
              height: 10,
            ),
          ],
        ),
      ),
    ),
  );
}
