import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../main.dart';
import '../util/color_utils.dart';
import '../widgets/widget.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({Key? key}) : super(key: key);

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  TextEditingController _passwordTextController = TextEditingController();
  TextEditingController _emailTextController = TextEditingController();
  TextEditingController _userNameTextController = TextEditingController();
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          elevation: 0,
          title: const Text(
            "Sign Up",
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
        ),
        body: Container(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.height,
            decoration: BoxDecoration(
                gradient: LinearGradient(colors: [
                  hexStringtoColor("CB2B93"),
                  hexStringtoColor("9546C4"),
                  hexStringtoColor("5E61F4")
                ], begin: Alignment.topCenter, end: Alignment.bottomCenter)),
            child: SingleChildScrollView(
                child: Padding(
                  padding: EdgeInsets.fromLTRB(20, 120, 20, 0),
                  child: Column(
                    children: <Widget>[
                      const SizedBox(
                        height: 20,
                      ),
                      reusableTextField("Enter UserName", Icons.person_outline, false,
                          _userNameTextController),
                      const SizedBox(
                        height: 20,
                      ),
                      reusableTextField("Enter Email Id", Icons.person_outline, false,
                          _emailTextController),
                      const SizedBox(
                        height: 20,
                      ),
                      reusableTextField("Enter Password", Icons.lock_outlined, true,
                          _passwordTextController),
                      const SizedBox(
                        height: 20,
                      ),
                      firebaseUIButton(context, "Sign Up", () {
                        FirebaseAuth.instance
                            .createUserWithEmailAndPassword(
                            email: _emailTextController.text,
                            password: _passwordTextController.text)
                            .then((value) {
                          print("Created New Account");
                          Navigator.popAndPushNamed(context, MyApp.Main_Screen);
                          // Navigator.push(context,
                          //     MaterialPageRoute(builder: (context) => HomeScreen()));
                        }).onError((error, stackTrace) {
                          print("Error ${error.toString()}");
                        });
                      })
                    ],
                  ),
                ))),
      ),
    );
  }
}
