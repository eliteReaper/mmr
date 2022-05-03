import 'dart:async';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:mmr_app/main.dart';

class SplashScreen extends StatelessWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Timer timer = Timer(const Duration(seconds: 2), (){
      Navigator.popAndPushNamed(context, MyApp.Login_Screen);
    });
    timer.tick;
    return const SafeArea(
      child: Scaffold(
        body: Text("Splash screen only"),
      ),
    );
  }
}
