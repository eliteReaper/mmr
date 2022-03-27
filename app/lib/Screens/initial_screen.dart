import 'dart:async';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:mmr_app/main.dart';

class InitialScreen extends StatefulWidget {
  const InitialScreen({Key? key}) : super(key: key);

  @override
  State<InitialScreen> createState() => _InitialScreenState();
}

class _InitialScreenState extends State<InitialScreen> {
  @override
  Widget build(BuildContext context) {
    Timer timer = Timer(const Duration(seconds: 5), (){
      Navigator.popAndPushNamed(context, MyApp.Login_Screen);
    });
    timer.tick;
    return const Scaffold(
      body: SafeArea(
        child: Text("Hello there init screen"),
      ),
    );
  }
}
