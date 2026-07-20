# Shell permission migration

The old generic command tool was removed. Polaris intentionally does not copy an old allow
rule to both dialects because that would widen permissions.

Convert each rule according to the syntax it authorizes:

```toml
[permissions]
allow = ["bash(git *)", "powershell(Get-ChildItem *)"]
deny = ["bash(rm *)", "powershell(Remove-Item *)"]
ask = ["bash", "powershell"]
```

If a legacy rule is found in user, project, explicit or managed configuration, startup
fails and prints the offending rule. Update the source configuration and retry.
