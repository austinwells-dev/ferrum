# Record with: vhs docs/media/ferrum-tui.tape   (run from the repo root)
# Requires: brew install vhs, and `cargo build --release --bin ferrum-tui`
Output docs/media/ferrum-tui.gif

Set Shell "zsh"
Set FontSize 16
Set Width 1200
Set Height 720
Set Padding 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms

Hide
Type "export PATH=$PWD/target/release:$PATH && clear" Enter
Show

Type "ferrum-tui" Enter
Sleep 6s            # startup animation

