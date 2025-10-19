from flask import Flask, jsonify, request, Response
import yaml

from db import list_strategies, load_strategy, save_strategy, remove_strategy

# Flask app
app = Flask(__name__)


@app.route("/strategies", methods=["GET"])
def get_strategies():
    """Return all strategies stored in the database as YAML.

    Query params:
      - name: optional, return only the named strategy

    Response: JSON object with keys: names (list) and strategies (dict name->yaml)
    If `name` is provided and not found, returns 404.
    """
    name = request.args.get("name")

    try:
        if name:
            # Load single strategy by name
            strategy = load_strategy(name)
            yaml_text = yaml.safe_dump(strategy)
            return Response(yaml_text, mimetype="text/yaml")

        # No name provided: list all strategies and return a JSON summary
        strategies_yaml = {}
        for n in list_strategies():
            try:
                s = load_strategy(n)
                strategies_yaml[n] = yaml.safe_dump(s)
            except Exception:
                strategies_yaml[n] = None

        return jsonify({"strategies": strategies_yaml})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "internal server error", "detail": str(e)}), 500


@app.route("/strategies", methods=["POST"])
def update_main_strategy():
    """Update a strategy to be the main in the database and remove all
    alternative strategies.

    Request body: Strategy name as plain text.

    Response: JSON object with key: message (success or error)
    """
    try:
        name = request.data.decode("utf-8").strip()
        if not name:
            return jsonify({"error": "strategy name is required"}), 400

        # Load the strategy to ensure it exists
        strategy = load_strategy(name)
        if not strategy:
            return jsonify({"error": f"strategy '{name}' not found"}), 404

        # Set this strategy as main and remove alternatives
        save_strategy("default_strategy", strategy)
        for n in list_strategies():
            if n != "default_strategy":
                remove_strategy(n)

        return jsonify({"message": f"strategy '{name}' set as main successfully"})
    except Exception as e:
        return jsonify({"error": "internal server error", "detail": str(e)}), 500


if __name__ == "__main__":
    # Simple runner for local development
    app.run(host="0.0.0.0", port=5000, debug=True)
