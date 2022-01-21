from twitchio import websocket
from twitchio.ext import commands
from twitchio.ext import routines
from twitchio.client import Client

class Bot(commands.Bot):

	def __init__(self):
		# Initialise our Bot with our access token, prefix and a list of channels to join on boot...
		# prefix can be a callable, which returns a list of strings or a string...
		# initial_channels can also be a callable which returns a list of strings...
		super().__init__(token="oauth:jxto1bp37wsrjv9djerjfyvy9a2881", prefix='!', initial_channels=['BirdBotML'])
	
	async def event_ready(self):
		# Notify us when everything is ready!
		# We are logged in and ready to chat and use commands...
		print(f'Logged in as | {self.nick}')
	
	async def event_message(self, message):
		# Messages with echo set to True are messages sent by the bot...
		# For now we just want to ignore them...
		if message.echo:
			return

		# Print the contents of our message to console...
		print(message.content)

		# Since we have commands and are overriding the default `event_message`
		# We must let the bot know we want to handle and invoke our commands...
		await self.handle_commands(message)

	@commands.command(name="hello")
	async def hello(self, ctx: commands.Context):
		# Here we have a command hello, we can invoke our command with our prefix and command name
		# e.g ?hello
		# We can also give our commands aliases (different names) to invoke with.

		# Send a hello back!
		# Sending a reply back to the channel is easy... Below is an example.
		await ctx.send(f'Hello {ctx.author.name}!')
		

@routines.routine(seconds=5.0, iterations=5)
async def hello(string: str):

	print(f'Hello {string}!')
	
hello.start("!hello")

if __name__ == '__main__':
	bot = Bot()
	bot.run()
	# bot.run() is blocking and will stop execution of any below code here until stopped or closed.
	ws = bot._ws
	ws.connect()
	ws.send_privmsg('BirdBotML', f"/I have landed")
	# THIS DID NOT RUN!
	print("Bot RUNNING!")