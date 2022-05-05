import 'dart:async';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:mmr_app/main.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Timer timer = Timer(const Duration(seconds: 3), () {
      Navigator.popAndPushNamed(context, MyApp.Login_Screen);
    });
    timer.tick;
    return SafeArea(
      child: Scaffold(
        body: Container(
          child: Column(
            children: [
              SizedBox(height: 70,),
              const Center(
                child: Text(
                  "Multimedia Recommender\nSystem",
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 28),
                  softWrap: true,
                ),
              ),
              // Expanded(child: Container()),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: const [
                    SpinKitWave(color: Colors.white,),
                  ],
                ),
              ),
              const SizedBox(
                height: 60,
              ),
            ],
          ),
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topRight,
              end: Alignment.bottomLeft,
              colors: [
                Colors.blue,
                Colors.red,
              ],
            ),
          ),
        ),
      ),
    );
  }
}
