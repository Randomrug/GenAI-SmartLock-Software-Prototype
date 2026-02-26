from flask import Blueprint, request, jsonify
import sqlite3

telegram_webhook = Blueprint('telegram_webhook', __name__)

def update_event_with_feedback(event_id, feedback):
    conn = sqlite3.connect('smart_lock_events.db')
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE access_events SET owner_feedback = ? WHERE id = ?
    """, (feedback, event_id))
    conn.commit()
    conn.close()

@telegram_webhook.route('/telegram_feedback', methods=['POST'])
def receive_telegram_feedback():
    data = request.json
    event_id = data.get('event_id')
    feedback = data.get('feedback')
    if not event_id or not feedback:
        return jsonify({'success': False, 'error': 'Missing event_id or feedback'}), 400
    update_event_with_feedback(event_id, feedback)
    return jsonify({'success': True, 'message': 'Feedback stored successfully'})
